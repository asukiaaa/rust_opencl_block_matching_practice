extern crate image;
use image::{GenericImage, RgbImage};

fn get_gray_pixels(file_name: &str) -> (Vec<u8>, usize, usize) {
    let img = image::open(file_name).unwrap().grayscale();
    let w = img.width(); // / 2;
    let h = img.height(); // / 2;
    (img.raw_pixels(), img.width() as usize, img.height() as usize)
}

fn fill_vec(target_vec: &mut Vec<f32>, value: f32, start_x: usize, start_y: usize, fill_w: usize, fill_h: usize, line_w: usize, line_h: usize) {
    for i in start_y.. start_y + fill_h {
        if i >= line_h { continue }
        for j in start_x.. start_x + fill_w {
            if j >= line_w { continue }
            let index = i*line_w + j;
            target_vec[index] = value;
        }
    }
}

fn get_slice_pixels(pixels: &Vec<u8>, w: usize, h: usize, block_left: usize, block_top: usize, block_w: usize, block_h: usize) -> Vec<u8> {
    let mut sliced_pixels = vec!();
    for i in block_top..block_top + block_h {
        let start_index = w * i + block_left;
        // able to run without cloning?
        let part: Vec<u8> = pixels[start_index .. start_index + block_w].iter().cloned().collect();
        sliced_pixels.extend_from_slice(&part);
    }
    sliced_pixels
}

fn get_point_from_blocks(pixels1: &Vec<u8>, pixels2: &Vec<u8>, size: usize) -> f32 {
    let mut point: f32 = 0.0;
    for i in 0..size {
        point += (pixels1[i] as f32 - pixels2[i] as f32).abs();
    }
    point
}

fn get_diff_point(left_pixels: &Vec<u8>, right_pixels: &Vec<u8>, w: usize, h: usize, block_w: usize, block_h: usize, left_x: usize, left_y: usize, right_x: usize, right_y: usize) -> f32 {
    if left_x + block_w >= w || right_x + block_w >= w ||
    left_y + block_h >= h || right_y + block_h >= h {
        return std::f32::MAX;
    }
    let left_block = get_slice_pixels(left_pixels, w, h, left_x, left_y, block_w, block_h);
    let right_block = get_slice_pixels(right_pixels, w, h, right_x, right_y, block_w, block_h);
    get_point_from_blocks(&left_block, &right_block, block_w * block_h)
}

fn block_match(left_pixels: &Vec<u8>, right_pixels: &Vec<u8>, w: usize, h: usize, block_w: usize, block_h: usize, max_diff: usize) -> Vec<f32> {
    let mut diff_vec = vec![max_diff as f32; w * h];
    for i in 0..h {
        if i % block_h != 0 { continue } // For step_by
        for j in 0..w {
            if j % block_w != 0 { continue } // For step_by
            let mut min_diff_point = std::f32::MAX;
            let mut min_diff_index = max_diff;
            for k in 0..max_diff {
                let diff_point = get_diff_point(&left_pixels, &right_pixels, w, h, block_w, block_h, j+k, i, j, i);
                if diff_point < min_diff_point {
                    min_diff_point = diff_point;
                    min_diff_index = k;
                }
            }
            fill_vec(&mut diff_vec, min_diff_index as f32, j, i, block_w, block_h, w, h);
        }
    }
    diff_vec
}

fn normalize_result_pixels(pixels: &Vec<f32>, max_diff: usize) -> Vec<u8> {
    // result_mat * std::u8::MAX as f32 / (max_diff + 2) as f32;
    pixels.into_iter().map(|p| (p * std::u8::MAX as f32 / (max_diff + 2) as f32) as u8).collect()
}

fn hsv_to_rgb(h: u8, s: u8, v: u8) -> Vec<u8> {
    let hf = (h as f32 * 360. / std::u8::MAX as f32) / 60.;
    let sf = s as f32 / std::u8::MAX as f32;
    let vf = v as f32;
    let h_floor = hf.floor();
    let ff = hf - h_floor;
    let p = (vf * (1. - sf)) as u8;
    let q = (vf * (1. - sf * ff)) as u8;
    let t = (vf * (1. - sf * (1. - ff))) as u8;

    match h_floor as u8 {
        0 => vec![v, t, p],
        1 => vec![q, v, p],
        2 => vec![p, v, t],
        3 => vec![p, q, v],
        4 => vec![t, p, v],
        5 => vec![v, p, q],
        6 => vec![v, t, p],
        _ => vec![0, 0, 0],
    }
}

fn main() {
    // let left_image_file_name = "data/aloeL.jpg";
    // let right_image_file_name = "data/aloeR.jpg";
    let left_image_file_name = "data/left.png";
    let right_image_file_name = "data/right.png";
    let (left_pixels, width, height) = get_gray_pixels(&left_image_file_name);
    let (right_pixels, _, _) = get_gray_pixels(&right_image_file_name);
    let block_w = 11;
    let block_h = 11;
    let max_diff = width / 4;
    let result_pixels = block_match(&left_pixels, &right_pixels, width, height, block_w, block_h, max_diff);
    let result_pixels = normalize_result_pixels(&result_pixels, max_diff);
    let mut pixels = vec![];
    for p in result_pixels {
        pixels.extend(hsv_to_rgb(p as u8, 255, 255));
    }
    let result_image = RgbImage::from_raw(width as u32, height as u32, pixels).unwrap();
    let _saved = result_image.save("result.png");
}
