//! LanceDB storage: one table per index with a native BM25 full-text index on
//! `text` and brute-force / IVF vector search on `vector`.

use std::sync::Arc;

use anyhow::{Context, Result};
use lancedb::arrow::arrow_array::{
    self, cast::AsArray, types::Float32Type, Array, FixedSizeListArray, RecordBatch, StringArray,
    UInt32Array,
};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table};
use serde::Serialize;

use crate::chunker::Chunk;
use crate::paths::Paths;

pub fn table_name(index_id: &str) -> String {
    format!("idx_{index_id}")
}

fn schema(dim: usize) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("end_line", DataType::UInt32, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim as i32),
            false,
        ),
    ]))
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub score: f32,
    pub text: String,
    /// Full line range of the stored chunk when `text` was trimmed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_lines: Option<(u32, u32)>,
}

pub async fn connect(paths: &Paths) -> Result<Connection> {
    let uri = paths.lancedb_dir();
    lancedb::connect(&uri.to_string_lossy())
        .execute()
        .await
        .with_context(|| format!("opening LanceDB at {}", uri.display()))
}

pub struct ChunkStore {
    pub table: Table,
}

impl ChunkStore {
    pub async fn open(conn: &Connection, index_id: &str) -> Result<Option<Self>> {
        match conn.open_table(table_name(index_id)).execute().await {
            Ok(table) => Ok(Some(Self { table })),
            Err(lancedb::Error::TableNotFound { .. }) => Ok(None),
            Err(e) => Err(e).context("opening table"),
        }
    }

    pub async fn open_or_create(conn: &Connection, index_id: &str, dim: usize) -> Result<Self> {
        if let Some(s) = Self::open(conn, index_id).await? {
            return Ok(s);
        }
        let table = conn
            .create_empty_table(table_name(index_id), schema(dim))
            .execute()
            .await
            .context("creating table")?;
        Ok(Self { table })
    }

    pub async fn drop(conn: &Connection, index_id: &str) -> Result<()> {
        match conn.drop_table(table_name(index_id), &[]).await {
            Ok(()) | Err(lancedb::Error::TableNotFound { .. }) => Ok(()),
            Err(e) => Err(e).context("dropping table"),
        }
    }

    pub async fn count(&self) -> Result<usize> {
        Ok(self.table.count_rows(None).await?)
    }

    /// Delete all chunks belonging to the given relative paths.
    pub async fn delete_paths(&self, paths: &[String]) -> Result<()> {
        for batch in paths.chunks(256) {
            let list = batch
                .iter()
                .map(|p| format!("'{}'", p.replace('\'', "''")))
                .collect::<Vec<_>>()
                .join(",");
            self.table
                .delete(format!("path IN ({list})").as_str())
                .await
                .context("deleting stale chunks")?;
        }
        Ok(())
    }

    pub async fn add(&self, chunks: &[Chunk], vectors: Vec<Vec<f32>>, dim: usize) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        let path = StringArray::from_iter_values(chunks.iter().map(|c| c.rel_path.as_str()));
        let start = UInt32Array::from_iter_values(chunks.iter().map(|c| c.start_line));
        let end = UInt32Array::from_iter_values(chunks.iter().map(|c| c.end_line));
        let text = StringArray::from_iter_values(chunks.iter().map(|c| c.text.as_str()));
        let vector = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vectors.into_iter().map(|v| Some(v.into_iter().map(Some))),
            dim as i32,
        );
        let batch = RecordBatch::try_new(
            schema(dim),
            vec![Arc::new(path), Arc::new(start), Arc::new(end), Arc::new(text), Arc::new(vector)],
        )?;
        self.table.add(batch).execute().await.context("adding chunks")?;
        Ok(())
    }

    pub async fn has_fts_index(&self) -> bool {
        self.table
            .list_indices()
            .await
            .map(|ix| ix.iter().any(|i| i.columns.iter().any(|c| c == "text")))
            .unwrap_or(false)
    }

    /// (Re)build the BM25 index on `text` plus a BTree on `path` for fast deletes.
    pub async fn rebuild_indices(&self) -> Result<()> {
        let rows = self.count().await?;
        if rows == 0 {
            return Ok(());
        }
        // Code-friendly tokenisation: no stemming / stop-word removal, keep
        // identifiers like `UCharacterMovementComponent` intact (lower-cased).
        let fts = FtsIndexBuilder::default().stem(false).remove_stop_words(false);
        self.table
            .create_index(&["text"], Index::FTS(fts))
            .replace(true)
            .execute()
            .await
            .context("building full-text index")?;
        let has_path_ix = self
            .table
            .list_indices()
            .await
            .map(|ix| ix.iter().any(|i| i.columns.iter().any(|c| c == "path")))
            .unwrap_or(false);
        if !has_path_ix {
            if let Err(e) = self.table.create_index(&["path"], Index::Auto).execute().await {
                tracing::warn!("could not create path index: {e}");
            }
        }
        if rows >= 100_000 {
            let has_vec_ix = self
                .table
                .list_indices()
                .await
                .map(|ix| ix.iter().any(|i| i.columns.iter().any(|c| c == "vector")))
                .unwrap_or(false);
            if !has_vec_ix {
                if let Err(e) = self.table.create_index(&["vector"], Index::Auto).execute().await {
                    tracing::warn!("could not create vector index: {e}");
                }
            }
        }
        if let Err(e) = self.table.optimize(lancedb::table::OptimizeAction::All).await {
            tracing::warn!("optimize failed: {e}");
        }
        Ok(())
    }

    /// Hybrid search: vector similarity + BM25, fused with Reciprocal Rank Fusion.
    /// Falls back to pure vector search if no FTS index exists yet.
    pub async fn search(&self, query: &str, vector: &[f32], limit: usize) -> Result<Vec<SearchHit>> {
        let cols = ["path", "start_line", "end_line", "text"];
        let select = lancedb::query::Select::columns(&cols);
        let batches: Vec<RecordBatch> = if self.has_fts_index().await && !query.trim().is_empty() {
            self.table
                .query()
                .full_text_search(FullTextSearchQuery::new(query.to_string()))
                .select(select)
                .limit(limit)
                .nearest_to(vector)?
                .execute()
                .await?
                .try_collect()
                .await?
        } else {
            self.table
                .query()
                .select(select)
                .limit(limit)
                .nearest_to(vector)?
                .execute()
                .await?
                .try_collect()
                .await?
        };

        let mut hits = Vec::new();
        for b in &batches {
            let col = |n: &str| b.column_by_name(n).with_context(|| format!("missing column {n}"));
            let path = col("path")?.as_string::<i32>().clone();
            let start = col("start_line")?.as_primitive::<arrow_array::types::UInt32Type>().clone();
            let end = col("end_line")?.as_primitive::<arrow_array::types::UInt32Type>().clone();
            let text = col("text")?.as_string::<i32>().clone();
            let score: Vec<f32> = if let Some(s) = b.column_by_name("_relevance_score") {
                s.as_primitive::<Float32Type>().values().to_vec()
            } else if let Some(d) = b.column_by_name("_distance") {
                d.as_primitive::<Float32Type>().values().iter().map(|d| 1.0 / (1.0 + d)).collect()
            } else {
                vec![0.0; b.num_rows()]
            };
            for i in 0..b.num_rows() {
                if path.is_null(i) {
                    continue;
                }
                hits.push(SearchHit {
                    path: path.value(i).to_string(),
                    start_line: start.value(i),
                    end_line: end.value(i),
                    score: score[i],
                    text: text.value(i).to_string(),
                    chunk_lines: None,
                });
            }
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(limit);
        Ok(hits)
    }
}
