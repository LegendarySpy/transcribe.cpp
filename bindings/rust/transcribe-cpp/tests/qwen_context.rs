mod common;

use transcribe_cpp::{ExtSlot, Model, Qwen3AsrRunOptions, RunExtension, RunOptions};

#[test]
#[ignore = "requires TRANSCRIBE_SMOKE_MODEL pointing to a Qwen3-ASR GGUF"]
fn context_is_per_request_and_works_in_batches() {
    let (path, audio) = common::smoke_fixtures("qwen context").expect("model and audio fixtures");
    let model = Model::load(path).unwrap();
    assert_eq!(model.arch(), "qwen3_asr");
    assert!(model.accepts_ext(
        ExtSlot::Run,
        transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_QWEN3_ASR_RUN
    ));
    assert!(!model.accepts_ext(
        ExtSlot::Stream,
        transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_QWEN3_ASR_RUN
    ));
    let mut session = model.session().unwrap();
    let plain = RunOptions {
        language: Some("en".into()),
        ..Default::default()
    };
    let baseline = session.run(&audio, &plain).unwrap().text;
    assert!(!baseline.is_empty());
    let with_context = |context: &str| RunOptions {
        family: Some(RunExtension::Qwen3Asr(Qwen3AsrRunOptions {
            context: Some(context.into()),
        })),
        ..plain.clone()
    };
    assert_eq!(
        session.run(&audio, &with_context("")).unwrap().text,
        baseline
    );
    let hints = with_context("Kennedy, Americans, Glimpse, José, 東京.");
    assert!(!session.run(&audio, &hints).unwrap().text.is_empty());
    let batch = session.run_batch(&[&audio, &audio], &hints).unwrap();
    assert!(
        batch
            .iter()
            .all(|r| r.as_ref().is_ok_and(|r| !r.text.is_empty()))
    );
    assert!(
        session
            .run(&audio, &with_context(&"x".repeat(4097)))
            .is_err()
    );
    assert!(session.run(&audio, &with_context("bad\0context")).is_err());
    assert_eq!(session.run(&audio, &plain).unwrap().text, baseline);
}
