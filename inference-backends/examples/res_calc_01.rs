use inference_backends::stablediffusioncpp::helpers::Rescalc;

fn main() {
    // DIN A4 ist 210mm × 297mm
    println!(
        "{:#?}",
        Rescalc::fit_first(210.0 / 297.0, 1024.0 * 1024.0, 0.95, 0.96)
    );
}
