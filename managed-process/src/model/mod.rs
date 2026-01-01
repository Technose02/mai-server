use core::fmt::Debug as TDebug;
use std::process::ExitStatus;
pub use tokio::sync::oneshot::Sender as OneShotSender;

pub enum ProcessProtocol<ProcessConfig>
where
    ProcessConfig: Clone + PartialEq + TDebug + Send + 'static,
{
    // sent by process
    ProcessStarted,
    ProcessFinished(Option<ExitStatus>),

    // sent by controller
    StartProcess(ProcessConfig),
    StopProcess,
    ReadProcessState(OneShotSender<ProcessState<ProcessConfig>>),
}

#[derive(Clone, Debug)]
pub enum ProcessState<ProcessConfig>
where
    ProcessConfig: Clone + PartialEq + TDebug + Send + 'static,
{
    // the process has not stopped or not started yet
    Stopped,

    // process is stopping ; provides current config as well as optionally the next config to use (if set this state will transition to starting with the next_config, else it will transition to stopped)
    Stopping(ProcessConfig, Option<ProcessConfig>),

    // process is running with given configuration
    Running(ProcessConfig),

    // process is starting using given configuration
    Starting(ProcessConfig),
}
