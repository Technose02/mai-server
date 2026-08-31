#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub struct Res {
    pub width: u32,
    pub height: u32,
}

pub struct Rescalc;

impl Rescalc {
    fn fit(target_ar: f32, target_size: f32, tol_ar: f32, tol_size: f32) -> Vec<(Res, f32, f32)> {
        let mut out = Vec::new();

        let min_tolerated_ar = (1.0 - tol_ar) * target_ar;
        let max_tolerated_ar = (1.0 + tol_ar) * target_ar;
        let min_tolerated_size = (1.0 - tol_size) * target_size;
        let max_tolerated_size = (1.0 + tol_size) * target_size;

        let prod1616 = 16.0 * 16.0;
        let i_max = (max_tolerated_size / prod1616).ceil() as u32;

        for i in 1..=i_max {
            let prod1616i = prod1616 * i as f32;

            let j_min = (min_tolerated_size / prod1616i).floor() as u32;
            let j_max = (max_tolerated_size / prod1616i).ceil() as u32;

            for j in j_min..=j_max {
                let ar = i as f32 / j as f32;
                if ar >= min_tolerated_ar && ar <= max_tolerated_ar {
                    let w = 16 * i;
                    let h = 16 * j;

                    let err_ar = (ar - target_ar) / target_ar;
                    let err_size = ((w * h) as f32 - target_size) / target_size;

                    out.push((
                        Res {
                            width: w,
                            height: h,
                        },
                        err_ar,
                        err_size,
                    ));
                }
            }
        }

        out.sort_by(|r1, r2| {
            let e1 = r1.1.abs() + r1.2.abs();
            let e2 = r2.1.abs() + r2.2.abs();
            //let e1 = std::cmp::max(r1.1.round().abs() as u32, r1.2.round().abs() as u32);
            //let e2 = std::cmp::max(r2.1.round().abs() as u32, r2.2.round().abs() as u32);
            if e1 < e2 {
                std::cmp::Ordering::Less
            } else if e1 > e2 {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });

        out
    }

    pub fn fit_first(
        target_ar: f32,
        target_size: f32,
        tol_ar: f32,
        tol_size: f32,
    ) -> Option<(Res, f32, f32)> {
        let res = Self::fit(target_ar, target_size, tol_ar, tol_size);
        res.into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test01() {
        let target_ar = 210.0 / 297.0;
        let tol_ar = 0.05;
        let target_size = 2.0 * 1024.0 * 1024.0;
        let tol_size = 0.05;

        let fit_results = Rescalc::fit(target_ar, target_size, tol_ar, tol_size);
        for (res, err_ar, err_size) in fit_results.iter() {
            println!(
                "Res: {} x {}\t-> err_ar: {err_ar}, err_size: {err_size}",
                res.width, res.height
            );
        }
        let best = fit_results.first().map(|r| r.0).unwrap();
        assert_eq!(
            best,
            Rescalc::fit_first(target_ar, target_size, tol_ar, tol_size)
                .unwrap()
                .0
        );
        println!("best: {}x{}", best.width, best.height);
    }
}
