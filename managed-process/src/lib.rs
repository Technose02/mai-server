mod backendcontroller;
mod model;
mod statemanager;

pub use backendcontroller::BackendController;
pub use model::ProcessProtocol;
pub use model::ProcessState;
pub(crate) use statemanager::ProcessStateManager;

pub trait RunBackendProcess {
    type ProcessConfig: Clone + PartialEq + core::fmt::Debug + Send + 'static;

    fn run_backend_process(
        &self,
        process_config: Self::ProcessConfig,
        cancel_receiver: tokio::sync::oneshot::Receiver<bool>,
        notifier: tokio::sync::mpsc::Sender<ProcessProtocol<Self::ProcessConfig>>,
    );
}
