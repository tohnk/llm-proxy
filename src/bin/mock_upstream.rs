use clap::Parser;
use llm_proxy::mock::{create_mock_app, MockServerState};
use tracing::info;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Mock OpenAI/Gemini upstream server for testing llm-proxy"
)]
struct Args {
    #[arg(short, long, default_value = "127.0.0.1:8788")]
    bind: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    tracing_subscriber::fmt::init();

    let state = MockServerState::default();
    let app = create_mock_app(state);

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    info!(
        "Mock upstream listening on http://{}",
        listener.local_addr()?
    );

    axum::serve(listener, app).await?;
    Ok(())
}
