//! End-to-end: index a tiny tree with a small model, search, then sync changes.
//! Downloads `bge-small-en-v1.5` (~130 MB) on first run, hence `#[ignore]`.

use std::sync::{Arc, Mutex};

use rag_core::{Engine, IndexConfig, Paths, Phase, SyncOptions};

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn index_search_and_incremental_sync() {
    let data = tempfile::tempdir().unwrap();
    let src = tempfile::tempdir().unwrap();
    let models = std::env::var("RAG_TEST_MODELS").ok();
    let engine = Engine::with_paths(Paths::at(data.path()).unwrap());
    #[cfg(unix)]
    if let Some(m) = models {
        // Reuse a model cache across test runs.
        std::fs::remove_dir_all(engine.paths.models_dir()).ok();
        std::os::unix::fs::symlink(m, engine.paths.models_dir()).unwrap();
    }

    let s = src.path();
    std::fs::create_dir_all(s.join("Source/Player")).unwrap();
    std::fs::create_dir_all(s.join("Intermediate")).unwrap();
    std::fs::write(
        s.join("Source/Player/HealthComponent.cpp"),
        "#include \"HealthComponent.h\"\n\nvoid UHealthComponent::ApplyDamage(float Amount)\n{\n    Health = FMath::Max(0.f, Health - Amount);\n    if (Health <= 0.f) { OnDeath.Broadcast(); }\n}\n\nvoid UHealthComponent::Heal(float Amount)\n{\n    Health = FMath::Min(MaxHealth, Health + Amount);\n}\n",
    )
    .unwrap();
    std::fs::write(
        s.join("Source/Player/Inventory.cpp"),
        "void UInventory::AddItem(const FItem& Item)\n{\n    Items.Add(Item);\n    OnInventoryChanged.Broadcast();\n}\n",
    )
    .unwrap();
    std::fs::write(s.join("Intermediate/Gen.cpp"), "void Generated() {}\n").unwrap();

    engine
        .save_index(
            IndexConfig {
                id: "test".into(),
                name: "Test".into(),
                root_path: s.to_string_lossy().into(),
                embed_model: "BAAI/bge-small-en-v1.5".into(),
                allowed_extensions: vec![".cpp".into(), ".h".into()],
                ignore_dirs: vec!["Intermediate".into()],
                chunk_size: 400,
                chunk_overlap: 40,
                auto_update_cooldown_hours: 0.0,
                ..Default::default()
            },
            None,
        )
        .unwrap();

    let phases = Arc::new(Mutex::new(Vec::new()));
    let ph = phases.clone();
    let r = engine
        .sync(
            "test",
            SyncOptions {
                progress: Some(Arc::new(move |p| ph.lock().unwrap().push(p.phase))),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(r.files_added, 2);
    assert!(r.total_chunks >= 2);
    assert!(phases.lock().unwrap().contains(&Phase::Embedding));

    let out = engine.search("test", "take damage and die", 3, false).await.unwrap();
    assert_eq!(out.hits[0].path, "Source/Player/HealthComponent.cpp", "{:#?}", out.hits);
    assert!(out.hits[0].start_line >= 1);
    assert!(out.notes.iter().all(|n| !n.contains("reranker")), "{:?}", out.notes);

    // No changes -> nothing embedded.
    let r = engine.sync("test", SyncOptions::default()).await.unwrap();
    assert_eq!(r.chunks_embedded, 0);

    // Modify + delete; auto-sync on search picks it up (cooldown 0).
    std::fs::remove_file(s.join("Source/Player/Inventory.cpp")).unwrap();
    std::fs::write(
        s.join("Source/Player/HealthComponent.cpp"),
        "void UHealthComponent::Regenerate(float DeltaTime)\n{\n    Health += RegenRate * DeltaTime;\n}\n",
    )
    .unwrap();
    let out = engine.search("test", "health regeneration over time", 5, true).await.unwrap();
    let r = out.auto_sync.expect("auto sync ran");
    assert_eq!((r.files_modified, r.files_removed), (1, 1));
    assert!(out.hits.iter().all(|h| h.path != "Source/Player/Inventory.cpp"));
    assert!(out.hits[0].text.contains("Regenerate"));

    let cfg = engine.config().unwrap();
    assert_eq!(cfg.index("test").unwrap().chunk_count as usize, r.total_chunks);
    assert!(cfg.reranker_model.is_some(), "reranker on by default");

    // Capped auto-sync: over the limit -> no sync, a note, stale-but-valid results.
    engine.update_settings(|c| c.auto_sync_max_files = 0).unwrap();
    std::fs::write(s.join("Source/Player/New.cpp"), "void UNew::Spawn() {}\n").unwrap();
    let out = engine.search("test", "regenerate health", 5, true).await.unwrap();
    assert!(out.auto_sync.is_none());
    assert!(out.notes.iter().any(|n| n.contains("not auto-updated")), "{:?}", out.notes);
    assert!(!out.hits.is_empty());

    // Trimming: long chunk -> window around the matching line.
    engine.update_settings(|c| c.snippet_max_lines = 5).unwrap();
    let mut long = String::new();
    for i in 0..60 {
        long.push_str(&format!("    int Unrelated{i} = {i};\n"));
    }
    long.push_str("    float StaminaDrainPerSecond = 12.f;\n");
    std::fs::write(s.join("Source/Player/Stamina.cpp"), format!("void UStamina::Init()\n{{\n{long}}}\n")).unwrap();
    engine.sync("test", SyncOptions::default()).await.unwrap();
    let out = engine.search("test", "StaminaDrainPerSecond", 1, false).await.unwrap();
    let h = &out.hits[0];
    assert_eq!(h.path, "Source/Player/Stamina.cpp");
    assert!(h.text.lines().count() <= 5, "{}", h.text);
    assert!(h.text.contains("StaminaDrainPerSecond"));

    engine.delete_index("test").await.unwrap();
    assert!(engine.config().unwrap().indexes.is_empty());
}
