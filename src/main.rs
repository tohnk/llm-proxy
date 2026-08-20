use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use llm_proxy::config::Config;
use llm_proxy::create_app;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Local resilient LLM proxy for Jan.ai and OpenAI-compatible upstreams"
)]
struct Args {
    #[arg(short, long, help = "Path to TOML configuration file")]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let config = match Config::load(args.config.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Configuration error: {e}");
            std::process::exit(1);
        }
    };

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log.level));

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!(
        bind = %config.server.bind,
        upstream = %config.upstream.base_url,
        resume_enabled = config.resume.enabled,
        resume_mode = ?config.resume.mode,
        max_attempts = config.resume.max_attempts,
        "Starting llm-proxy"
    );

    let bind_addr = config.server.bind.clone();
    let listener = TcpListener::bind(&bind_addr).await.map_err(|e| {
        error!(error = %e, bind = %bind_addr, "Failed to bind server address");
        e
    })?;

    info!("Proxy listening on http://{}", listener.local_addr()?);

    let app = create_app(Arc::new(config));

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Proxy shut down cleanly");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
