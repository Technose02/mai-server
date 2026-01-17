use crate::{ProcessState, ProcessStateManager, RunBackendProcess, model::ProcessProtocol};
use core::fmt::Debug as TDebug;
use tokio::{
    spawn,
    sync::{
        mpsc::{Sender as MpscSender, channel as mpsc_channel},
        oneshot::channel as oneshot_channel,
    },
};

#[derive(Clone, Debug)]
pub struct BackendController<ProcessConfig>(MpscSender<ProcessProtocol<ProcessConfig>>)
where
    ProcessConfig: Clone + PartialEq + TDebug + Send + 'static;

impl<ProcessConfig> From<MpscSender<ProcessProtocol<ProcessConfig>>>
    for BackendController<ProcessConfig>
where
    ProcessConfig: Clone + PartialEq + TDebug + Send + 'static,
{
    fn from(controller_sender: MpscSender<ProcessProtocol<ProcessConfig>>) -> Self {
        Self(controller_sender)
    }
}

impl<ProcessConfig> BackendController<ProcessConfig>
where
    ProcessConfig: Clone + PartialEq + TDebug + Send + 'static,
{
    pub async fn init_backend<Backend>(backend: Backend) -> BackendController<ProcessConfig>
    where
        Backend: RunBackendProcess<ProcessConfig = ProcessConfig> + Send + 'static,
    {
        let (controller_sender, mut controller_receiver) =
            mpsc_channel::<ProcessProtocol<ProcessConfig>>(1);

        let mut state_manager =
            ProcessStateManager::<Backend, ProcessConfig>::new(backend, controller_sender);

        let ret = BackendController::from(state_manager.controller_sender());
        spawn(async move {
            loop {
                match controller_receiver.recv().await {
                    Some(ProcessProtocol::<ProcessConfig>::ProcessFinished(
                        optional_exit_status,
                    )) => state_manager.on_process_finished(optional_exit_status),
                    Some(ProcessProtocol::<ProcessConfig>::ProcessStarted) => {
                        state_manager.on_process_started()
                    }
                    Some(ProcessProtocol::<ProcessConfig>::StartProcess((config, parallel))) => {
                        state_manager.on_start_process(config, parallel)
                    }
                    Some(ProcessProtocol::<ProcessConfig>::StopProcess) => {
                        state_manager.on_stop_process()
                    }
                    Some(ProcessProtocol::<ProcessConfig>::ReadProcessState(back_chan)) => {
                        state_manager.on_read_config(back_chan)
                    }
                    _ => {}
                }
            }
        });
        ret
    }

    pub async fn start(&self, config: ProcessConfig, parallel: u8) {
        self.0
            .send(ProcessProtocol::<ProcessConfig>::StartProcess((
                config, parallel,
            )))
            .await
            .unwrap();
    }
    pub async fn stop(&self) {
        self.0
            .send(ProcessProtocol::<ProcessConfig>::StopProcess)
            .await
            .unwrap();
    }
    pub async fn read_state(&self) -> ProcessState<ProcessConfig> {
        let (state_sender, state_receiver) = oneshot_channel::<ProcessState<ProcessConfig>>();
        self.0
            .send(ProcessProtocol::<ProcessConfig>::ReadProcessState(
                state_sender,
            ))
            .await
            .unwrap();
        state_receiver.await.unwrap()
    }
}
