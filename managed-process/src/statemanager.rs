use crate::{ProcessProtocol, ProcessState, RunBackendProcess};
use std::{
    fmt::Debug,
    process::ExitStatus,
    sync::{Arc, Mutex},
};
use tokio::sync::{
    mpsc::Sender as MpscSender,
    oneshot::{Sender as OneShotSender, channel as oneshot_channel},
};

pub struct ProcessStateManager<Backend, ProcessConfig>
where
    ProcessConfig: Clone + PartialEq + Debug + Send + 'static,
{
    backend: Backend,
    state: Option<ProcessState<ProcessConfig>>,
    optional_cancel_sender: Arc<Mutex<Option<OneShotSender<bool>>>>,
    controller_sender_proto: MpscSender<ProcessProtocol<ProcessConfig>>,
}

impl<Backend, ProcessConfig> ProcessStateManager<Backend, ProcessConfig>
where
    ProcessConfig: Clone + PartialEq + Debug + Send + 'static,
    Backend: RunBackendProcess<ProcessConfig = ProcessConfig> + Send + 'static,
{
    pub fn new(
        backend: Backend,
        controller_sender: MpscSender<ProcessProtocol<ProcessConfig>>,
    ) -> Self {
        Self {
            backend,
            state: Some(ProcessState::<ProcessConfig>::Stopped),
            optional_cancel_sender: Arc::new(Mutex::new(None)),
            controller_sender_proto: controller_sender,
        }
    }

    pub fn controller_sender(&self) -> MpscSender<ProcessProtocol<ProcessConfig>> {
        self.controller_sender_proto.clone()
    }

    pub fn on_process_finished(&mut self, _optional_exit_status: Option<ExitStatus>) {
        let state = self.state.take();

        match state {
            Some(ProcessState::<ProcessConfig>::Stopping(
                _,
                Some((next_config_handle, parallel)),
            )) => {
                let (cancel_sender, cancel_receiver) = oneshot_channel::<bool>();
                let mut ocs = self.optional_cancel_sender.lock().unwrap();
                *ocs = Some(cancel_sender);
                self.backend.run_backend_process(
                    next_config_handle.clone(),
                    parallel,
                    cancel_receiver,
                    self.controller_sender_proto.clone(),
                );
                self.state = Some(ProcessState::<ProcessConfig>::Starting((
                    next_config_handle,
                    parallel,
                )));
            }
            _ => self.state = Some(ProcessState::<ProcessConfig>::Stopped),
        }
    }

    pub fn on_process_started(&mut self) {
        if let Some(ProcessState::<ProcessConfig>::Starting((config, parallel))) = self.state.take()
        {
            self.state = Some(ProcessState::<ProcessConfig>::Running((config, parallel)));
        } else {
            panic!("received 'ProcessStarted' but state is not 'Starting'")
        }
    }

    pub fn on_start_process(&mut self, config: ProcessConfig, parallel: u8) {
        match &self.state {
            Some(ProcessState::<ProcessConfig>::Running((cur_config, p))) => {
                if *cur_config == config && *p == parallel {
                    return;
                }
            }
            Some(ProcessState::<ProcessConfig>::Starting((starting_config, p))) => {
                if *starting_config == config && *p == parallel {
                    return;
                }
            }
            Some(ProcessState::<ProcessConfig>::Stopping(_, Some((next_config, p)))) => {
                if *next_config == config && *p == parallel {
                    return;
                }
            }
            _ => {}
        }

        let state = self.state.take();
        match state {
            Some(ProcessState::<ProcessConfig>::Running((old_config, _))) => {
                self.state = Some(ProcessState::<ProcessConfig>::Stopping(
                    old_config,
                    Some((config, parallel)),
                ));

                if let Some(cancel_sender) = self.optional_cancel_sender.lock().unwrap().take() {
                    cancel_sender.send(true).unwrap();
                } else {
                    eprintln!("err: expected cancel_sender to be available but was None");
                }
            }
            Some(ProcessState::<ProcessConfig>::Starting((old_config, _))) => {
                self.state = Some(ProcessState::<ProcessConfig>::Stopping(
                    old_config,
                    Some((config, parallel)),
                ));
                if let Some(cancel_sender) = self.optional_cancel_sender.lock().unwrap().take() {
                    cancel_sender.send(true).unwrap();
                } else {
                    eprintln!("err: expected cancel_sender to be available but was None");
                }
            }
            Some(ProcessState::<ProcessConfig>::Stopping(old_config, _)) => {
                self.state = Some(ProcessState::<ProcessConfig>::Stopping(
                    old_config,
                    Some((config, parallel)),
                ));
            }
            Some(ProcessState::<ProcessConfig>::Stopped) => {
                let (cancel_sender, cancel_receiver) = oneshot_channel::<bool>();
                let mut ocs = self.optional_cancel_sender.lock().unwrap();
                *ocs = Some(cancel_sender);
                self.backend.run_backend_process(
                    config.clone(),
                    parallel,
                    cancel_receiver,
                    self.controller_sender(),
                );
                self.state = Some(ProcessState::<ProcessConfig>::Starting((config, parallel)));
            }
            _ => unreachable!(),
        }
    }

    pub fn on_stop_process(&mut self) {
        match &self.state {
            Some(ProcessState::Stopped) | Some(ProcessState::Stopping(_, None)) => return,
            _ => {}
        }
        let state = self.state.take();

        match state {
            Some(ProcessState::<ProcessConfig>::Running(cfg)) => {
                self.state = Some(ProcessState::<ProcessConfig>::Stopping(cfg.0, None));
                if let Some(cancel_sender) = self.optional_cancel_sender.lock().unwrap().take() {
                    cancel_sender.send(true).unwrap();
                }
            }
            Some(ProcessState::<ProcessConfig>::Starting(cfg)) => {
                self.state = Some(ProcessState::<ProcessConfig>::Stopping(cfg.0, None));
                if let Some(cancel_sender) = self.optional_cancel_sender.lock().unwrap().take() {
                    cancel_sender.send(true).unwrap();
                }
            }
            Some(ProcessState::<ProcessConfig>::Stopping(cfg, Some(_))) => {
                self.state = Some(ProcessState::<ProcessConfig>::Stopping(cfg, None));
            }
            _ => unreachable!(),
        }
    }

    pub fn on_read_config(&self, back_chan: OneShotSender<ProcessState<ProcessConfig>>) {
        back_chan.send(self.state.clone().unwrap()).unwrap();
    }
}
