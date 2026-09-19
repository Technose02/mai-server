#!/bin/bash

cargo run -p inference-backends --example zimage_turbo_01 1     > ex_zimage_turbo_01.log
cargo run -p inference-backends --example fluxdev 1             > ex_fluxdev.log
cargo run -p inference-backends --example krea2_turbo_01 1      > ex_krea2_turbo_01.log
cargo run -p inference-backends --example zimage_01 1           > ex_zimage_01.log
cargo run -p inference-backends --example krea2_turbo_02 1      > ex_krea2_turbo_02.log
cargo run -p inference-backends --example krea2_turbo_03 1      > ex_krea2_turbo_03.log
cargo run -p inference-backends --example anima_turbo_01 1      > ex_anima_turbo_01.log
cargo run -p inference-backends --example booguimage_turbo_01 1 > ex_booguimage_turbo_01.log
cargo run -p inference-backends --example flux2klein9b_01 1     > ex_flux2klein9b_01.log
cargo run -p inference-backends --example flux2klein9b_02 1     > ex_flux2klein9b_02.log
cargo run -p inference-backends --example flux2klein9b_03 1     > ex_flux2klein9b_03.log
cargo run -p inference-backends --example fluxschnell_01 1      > ex_fluxschnell_01.log
cargo run -p inference-backends --example mage_flow_turbo_01 1  > ex_mage_flow_turbo_01.log
