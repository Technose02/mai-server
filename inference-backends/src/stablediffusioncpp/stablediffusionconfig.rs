use crate::stablediffusioncpp::{StableDiffusionError, StableDiffusionJob};
use std::{path::PathBuf, time::Duration};
use tokio::{
    io::{AsyncReadExt, BufReader},
    sync::mpsc::{Receiver, channel},
    time::Instant,
};

pub struct StableDiffusionCppConfig {
    path_to_executable: PathBuf,
    temporary_output_dir: Option<PathBuf>,
    flash_attention_mode: FlashAttentionMode,
}

#[derive(Clone, Copy, Debug)]
pub enum FlashAttentionMode {
    All,
    DiffusionOnly,
    None,
}

pub enum StableDiffusionEvent {
    GenerationStarted {
        seed: u32,
        started_at: Instant,
    },
    Progress {
        step: usize,
        nsteps: usize,
        duration: Duration,
    },
    StdOutLine(String),
    StdErrLine(String),
    Error(StableDiffusionError),
    GenerationFinished {
        boxed_data: Box<Vec<u8>>,
        duration: Duration,
    },
}

impl StableDiffusionCppConfig {
    pub fn init(
        path_to_executable: impl Into<PathBuf>,
    ) -> core::result::Result<StableDiffusionCppConfig, StableDiffusionError> {
        let path_to_executable = path_to_executable.into();
        if !path_to_executable.is_file() {
            Err(StableDiffusionError::Custom(
                "invalid path to stablediffusion.cpp executable".to_string(),
            ))
        } else {
            //stable-diffusion.cpp version
            let mut cmd = std::process::Command::new(&path_to_executable);
            cmd.arg("--version");
            let output = cmd.output().map_err(|e| {
                StableDiffusionError::Custom(format!("Error running stablediffusion.cpp: {e}"))
            })?;
            if output
                .stdout
                .starts_with("stable-diffusion.cpp version".as_bytes())
            {
                Ok(StableDiffusionCppConfig {
                    path_to_executable,
                    temporary_output_dir: None,
                    flash_attention_mode: FlashAttentionMode::None,
                })
            } else {
                Err(StableDiffusionError::Custom(
                    "this is not stablediffusion.cpp".to_string(),
                ))
            }
        }
    }

    pub fn with_temporary_output_dir(mut self, temporary_output_dir: impl Into<PathBuf>) -> Self {
        self.temporary_output_dir = Some(temporary_output_dir.into());
        self
    }

    pub fn with_flash_attention_mode(mut self, flash_attention_mode: FlashAttentionMode) -> Self {
        self.flash_attention_mode = flash_attention_mode;
        self
    }

    pub fn run<J: StableDiffusionJob>(
        &self,
        job: &J,
        //        event_sender: mpsc::Sender<StableDiffusionEvent>,
    ) -> Result<Receiver<StableDiffusionEvent>, StableDiffusionError> {
        let mut cmd = tokio::process::Command::new(&self.path_to_executable);

        let (event_sender, event_receiver) = channel::<StableDiffusionEvent>(1);

        let seed = job.seed().unwrap_or(rand::random::<u32>());

        let tmp_output = "sd_temp_out.png";
        let output_dir = if let Some(output_dir) = &self.temporary_output_dir {
            output_dir.clone()
        } else {
            std::env::current_dir().map_err(|e| {
                StableDiffusionError::Custom(format!(
                    "no temporary output dir provided and unable to retrieve working-dir: {}",
                    e
                ))
            })?
        };

        cmd.current_dir(&output_dir);
        cmd.arg("--diffusion-model").arg(job.diffusion_model());
        cmd.arg("--llm").arg(job.llm());
        cmd.arg("--vae").arg(job.vae());
        cmd.arg("--cfg-scale")
            .arg(format!("{:.1}", job.cfg_scale()));
        cmd.arg("--guidance").arg(format!("{:.1}", job.guidance()));
        cmd.arg("--steps").arg(job.steps().to_string());
        cmd.arg("--seed").arg(seed.to_string());
        cmd.arg("--width").arg(job.width().to_string());
        cmd.arg("--height").arg(job.height().to_string());
        cmd.arg("--prompt").arg(job.prompt());
        cmd.arg("--output").arg(tmp_output);
        cmd.arg("--scheduler").arg(job.scheduler());
        cmd.arg("--sampling-method").arg(job.sampling_method());

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        match self.flash_attention_mode {
            FlashAttentionMode::All => {
                cmd.arg("--fa");
            }
            FlashAttentionMode::DiffusionOnly => {
                cmd.arg("--diffusion-fa");
            }
            _ => {}
        }

        if job.vae_tiling() {
            cmd.arg("--vae-tiling");
        }

        if job.offload_to_cpu() {
            cmd.arg("--offload-to-cpu");
        }

        //println!("CMD: {:#?}", cmd);

        let mut child = cmd.spawn().expect("failed to spawn sd-cli process");
        let started_at = Instant::now();

        let mut out_reader = BufReader::new(child.stdout.take().unwrap());
        let mut err_reader = BufReader::new(child.stderr.take().unwrap());

        let out_sender = event_sender.clone();
        let err_sender = event_sender.clone();

        // Task Verarbeitung von stdout
        tokio::spawn(async move {
            let progress_regex = regex::Regex::new(r"\|[=>\s]+\|\s*(\d+)/(\d+)").unwrap();
            let mut buffer = [0u8; 1024];
            let mut current_line = Vec::new();

            loop {
                match out_reader.read(&mut buffer).await {
                    Ok(0) => break, // Stream beendet
                    Ok(n) => {
                        for &byte in &buffer[..n] {
                            // Bei \r oder \n interpretieren wir den bisherigen Text
                            if byte == b'\r' || byte == b'\n' {
                                if !current_line.is_empty() {
                                    if let Ok(text) = std::str::from_utf8(&current_line) {
                                        if let Err(e) = out_sender
                                            .send(StableDiffusionEvent::StdOutLine(
                                                text.to_string(),
                                            ))
                                            .await
                                        {
                                            eprintln!(
                                                "error sending StableDiffusionEvent::StdOutLine: {}",
                                                e
                                            );
                                        }

                                        // Regex-Matching auf die extrahierte Zeile anwenden
                                        if let Some(captures) = progress_regex.captures(text) {
                                            // Capture-Gruppe 1: Aktueller Schritt
                                            let current = captures.get(1).unwrap().as_str();
                                            // Capture-Gruppe 2: Gesamtschritte
                                            let total = captures.get(2).unwrap().as_str();

                                            // Werte parsen und logisch verarbeiten
                                            if let (Ok(curr_num), Ok(total_num)) =
                                                (current.parse::<usize>(), total.parse::<usize>())
                                                && let Err(e) = out_sender
                                                    .send(StableDiffusionEvent::Progress {
                                                        step: curr_num,
                                                        nsteps: total_num,
                                                        duration: started_at.elapsed(),
                                                    })
                                                    .await
                                            {
                                                eprintln!(
                                                    "error sending StableDiffusionEvent::Progress: {}",
                                                    e
                                                );
                                            }
                                        }
                                    }
                                    current_line.clear();
                                }
                            } else {
                                current_line.push(byte);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Fehler beim Lesen des Streams: {}", e);
                        break;
                    }
                }
            }
        });

        // Task Verarbeitung von stderr
        tokio::spawn(async move {
            let mut buffer = [0u8; 1024];
            let mut current_line = Vec::new();

            loop {
                match err_reader.read(&mut buffer).await {
                    Ok(0) => break, // Stream beendet
                    Ok(n) => {
                        for &byte in &buffer[..n] {
                            // Bei \r oder \n interpretieren wir den bisherigen Text
                            if byte == b'\r' || byte == b'\n' {
                                if !current_line.is_empty() {
                                    if let Ok(text) = std::str::from_utf8(&current_line)
                                        && let Err(e) = err_sender
                                            .send(StableDiffusionEvent::StdErrLine(
                                                text.to_string(),
                                            ))
                                            .await
                                    {
                                        eprintln!(
                                            "error sending StableDiffusionEvent::StdErrLine: {}",
                                            e
                                        );
                                    }
                                    current_line.clear();
                                }
                            } else {
                                current_line.push(byte);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Fehler beim Lesen des Streams: {}", e);
                        break;
                    }
                }
            }
        });

        // Task Warten auf Fertigstellung
        tokio::spawn(async move {
            if let Err(e) = event_sender
                .send(StableDiffusionEvent::GenerationStarted { seed, started_at })
                .await
            {
                eprintln!(
                    "error sending StableDiffusionEvent::GenerationStarted: {}",
                    e
                );
            }

            match child.wait().await {
                Ok(status) => {
                    if status.success() {
                        let image_file = output_dir.join(tmp_output);
                        let data = tokio::fs::read(&image_file).await.map_err(|e| {
                            StableDiffusionError::Custom(format!(
                                "unable to read from temp-imagefile '{}': '{e}'",
                                image_file.to_string_lossy()
                            ))
                        });

                        if let Err(error) = data {
                            if let Err(e) =
                                event_sender.send(StableDiffusionEvent::Error(error)).await
                            {
                                eprintln!("error sending StableDiffusionEvent::Error: {}", e);
                            }
                        } else {
                            // remove temporary file
                            if let Err(error) =
                                tokio::fs::remove_file(&image_file).await.map_err(|e| {
                                    StableDiffusionError::Custom(format!(
                                        "unable to remove temp-imagefile '{}': '{e}'",
                                        image_file.to_string_lossy()
                                    ))
                                })
                            {
                                if let Err(e) =
                                    event_sender.send(StableDiffusionEvent::Error(error)).await
                                {
                                    eprintln!("error sending StableDiffusionEvent::Error: {}", e);
                                }
                            } else {
                                // send result
                                if let Err(e) = event_sender
                                    .send(StableDiffusionEvent::GenerationFinished {
                                        boxed_data: Box::new(data.unwrap()),
                                        duration: started_at.elapsed(),
                                    })
                                    .await
                                {
                                    eprintln!(
                                        "error sending StableDiffusionEvent::GenerationFinished: {}",
                                        e
                                    );
                                }
                            }
                        }
                    } else {
                        if let Err(e) = event_sender
                            .send(StableDiffusionEvent::Error(StableDiffusionError::Custom(
                                format!(
                                    "stable-diffusion.cpp exited with status {}",
                                    status.code().unwrap_or(-1)
                                ),
                            )))
                            .await
                        {
                            eprintln!("error sending StableDiffusionEvent::Error: {}", e);
                        }
                    }
                }
                Err(error) => {
                    // send error as event
                    if let Err(e) = event_sender
                        .send(StableDiffusionEvent::Error(StableDiffusionError::Custom(
                            format!("error running stablediffusion.cpp: {error}"),
                        )))
                        .await
                    {
                        eprintln!("error sending StableDiffusionEvent::Error: {}", e);
                    }
                }
            }
        });

        Ok(event_receiver)
    }
}

#[cfg(test)]
mod test {

    use std::ops::Add;

    use super::*;

    // ROCM
    const VALID_PATH_TO_EXECUTABLE: &str =
        "/data0/inference/stable-diffusion.cpp/build-rocm/bin/sd-cli";

    // VULKAN
    //const VALID_PATH_TO_EXECUTABLE: &str =
    //    "/data0/inference/stable-diffusion.cpp/build-vulkan/bin/sd-cli";

    #[test]
    fn init_stable_diffusion_cpp_config_with_valid_path_works() {
        let path_to_executable: PathBuf = VALID_PATH_TO_EXECUTABLE.into();
        assert!(StableDiffusionCppConfig::init(path_to_executable).is_ok())
    }

    #[tokio::test]
    pub async fn run_krea2_job_works() {
        let path_to_executable: PathBuf = VALID_PATH_TO_EXECUTABLE.into();
        let sdcfg = StableDiffusionCppConfig::init(path_to_executable)
            .unwrap()
            .with_temporary_output_dir("/tmp")
            .with_flash_attention_mode(FlashAttentionMode::All);

        let job = crate::stablediffusioncpp::Krea2TurboJob::default()
            .with_width(1280)
            .with_height(720)
            .with_steps(12)
            .with_prompt("a cute little owl drinking coffee");

        let mut event_receiver = sdcfg.run(&job).unwrap();

        while let Some(event) = event_receiver.recv().await {
            match event {
                StableDiffusionEvent::GenerationStarted {
                    seed,
                    started_at: _,
                } => {
                    println!("generation job started with seed {seed}");
                }
                StableDiffusionEvent::Progress {
                    step,
                    nsteps,
                    duration,
                } => {
                    println!(
                        "still generating (step {step} of {nsteps} completed, {}ms elapsed)",
                        duration.as_millis()
                    )
                }
                StableDiffusionEvent::Error(e) => {
                    panic!("generation failed with an error: {e}");
                }
                StableDiffusionEvent::GenerationFinished {
                    boxed_data,
                    duration,
                } => {
                    tokio::fs::write("testimage.png", *boxed_data)
                        .await
                        .unwrap();
                    println!(
                        "generation job finished successfully after {} (see 'testimage.png')",
                        chrono::NaiveTime::from_hms_opt(0, 0, 0)
                            .unwrap()
                            .add(chrono::Duration::milliseconds(duration.as_millis() as i64))
                            .format("%H:%M:%S")
                    );
                    break;
                }

                // ignore other output for now
                StableDiffusionEvent::StdErrLine(_) | StableDiffusionEvent::StdOutLine(_) => {}
            }
        }
    }
}
