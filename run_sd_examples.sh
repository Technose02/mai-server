#!/bin/bash

rm *.png
rm ex*.log

echo "running example qwenimage21_01"
cargo run -p inference-backends --example qwenimage21_01 1          > ex_qwenimage21_01.log
echo "running example qwenimage21_02"
cargo run -p inference-backends --example qwenimage21_02 1          > ex_qwenimage21_02.log
echo "running example booguimageedit_turbo_01"
cargo run -p inference-backends --example booguimageedit_turbo_01 1 > ex_booguimageedit_turbo_01.log
echo "running example krea2_turbo_01"
cargo run -p inference-backends --example krea2_turbo_01 1          > ex_krea2_turbo_01.log
echo "running example zimage_turbo_01"
cargo run -p inference-backends --example zimage_turbo_01 1        > ex_zimage_turbo_01.log
echo "running example fluxdev"
cargo run -p inference-backends --example fluxdev 1                > ex_fluxdev.log
echo "running example zimage_01"
cargo run -p inference-backends --example zimage_01 1              > ex_zimage_01.log
echo "running example krea2_turbo_02"
cargo run -p inference-backends --example krea2_turbo_02 1         > ex_krea2_turbo_02.log
echo "running example krea2_turbo_03"
cargo run -p inference-backends --example krea2_turbo_03 1         > ex_krea2_turbo_03.log
echo "running example anima_turbo_01"
cargo run -p inference-backends --example anima_turbo_01 1         > ex_anima_turbo_01.log
echo "running example booguimage_turbo_01"
cargo run -p inference-backends --example booguimage_turbo_01 1    > ex_booguimage_turbo_01.log
echo "running example flux2klein9b_01"
cargo run -p inference-backends --example flux2klein9b_01 1        > ex_flux2klein9b_01.log
echo "running example flux2klein9b_02"
cargo run -p inference-backends --example flux2klein9b_02 1        > ex_flux2klein9b_02.log
echo "running example flux2klein9b_03"
cargo run -p inference-backends --example flux2klein9b_03 1        > ex_flux2klein9b_03.log
echo "running example fluxschnell_01"
cargo run -p inference-backends --example fluxschnell_01 1         > ex_fluxschnell_01.log
echo "running example mage_flow_turbo_01"
cargo run -p inference-backends --example mage_flow_turbo_01 1     > ex_mage_flow_turbo_01.log
