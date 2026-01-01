use managed_process::{BackendController, ProcessProtocol, RunBackendProcess};
use std::process::Stdio;
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
    spawn,
};

mod llamacppconfig;
pub use llamacppconfig::{LlamaCppConfig, LlamaCppConfigArgs};

pub type LlamaCppProtocol = ProcessProtocol<LlamaCppConfig>;
pub type LlamaCppBackendController = BackendController<LlamaCppConfig>;

pub struct LlamaCppBackend {
    pub host: String,
    pub port: u16,
    pub llama_cpp_command: String,
    pub llama_cpp_execdir: String,
}

impl RunBackendProcess for LlamaCppBackend {
    type ProcessConfig = LlamaCppConfig;

    fn run_backend_process(
        &self,
        process_config: Self::ProcessConfig,
        cancel_receiver: tokio::sync::oneshot::Receiver<bool>,
        notifier: tokio::sync::mpsc::Sender<ProcessProtocol<Self::ProcessConfig>>,
    ) {
        // prepare llama-cpp-command:
        let mut cmd = Command::new(&self.llama_cpp_command);
        cmd.current_dir(&self.llama_cpp_execdir);

        // set environment variables
        process_config.apply_env(&mut cmd);

        // set params
        cmd.arg("--host");
        cmd.arg(&self.host);

        cmd.arg("--port");
        cmd.arg(self.port.to_string());

        process_config.apply_args(&mut cmd);

        // provide std-streams
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped());

        cmd.kill_on_drop(true);

        // spawn process
        let mut proc_handle = cmd.spawn().unwrap();

        // spawn std-err observing task
        let stderr = proc_handle.stderr.take().unwrap();
        let notifier_cloned = notifier.clone();
        spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            loop {
                if let Ok(Some(line)) = lines.next_line().await
                    && line.contains("server is listening on http")
                {
                    notifier_cloned
                        .send(LlamaCppProtocol::ProcessStarted)
                        .await
                        .unwrap();
                }
            }
        });

        spawn(async move {
            tokio::select! {
                s = proc_handle.wait() => {
                    let exit_status = s.unwrap();
                    if exit_status.success() {
                        println!("llama-cpp-process ended successfully");
                        notifier.send(LlamaCppProtocol::ProcessFinished(None)).await.unwrap();
                    } else {
                        eprintln!("llama-cpp-process ended unsuccessfully with error exit_status {exit_status}");
                        notifier.send(LlamaCppProtocol::ProcessFinished(Some(exit_status))).await.unwrap();
                    }
                },
                _ = cancel_receiver => {
                    println!("killing llama-cpp-process");
                    proc_handle.kill().await.unwrap();
                    notifier.send(LlamaCppProtocol::ProcessFinished(None)).await.unwrap();
                }
            }
        });
    }
}
