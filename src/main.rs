extern crate image;
extern crate ocl;
use image::{GenericImage, RgbImage};
use ocl::{Buffer, MemFlags, ProQue, SpatialDims};

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

fn block_match(left_pixels: &Vec<u8>, right_pixels: &Vec<u8>, w: usize, h: usize, block_w: usize, block_h: usize, diff_len: usize) -> Vec<f32> {
    let mut diff_vec = vec![diff_len as f32; w * h];
    for i in 0..h {
        if i % block_h != 0 { continue } // For step_by
        for j in 0..w {
            if j % block_w != 0 { continue } // For step_by
            let mut min_diff_point = std::f32::MAX;
            let mut min_diff_index = diff_len;
            for k in 0..diff_len {
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

fn normalize_result_pixels(pixels: &Vec<f32>, diff_len: usize) -> Vec<u8> {
    // result_mat * std::u8::MAX as f32 / (diff_len + 2) as f32;
    pixels.into_iter().map(|p| (p * std::u8::MAX as f32 / (diff_len + 2) as f32) as u8).collect()
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
    let diff_len = width / 4;

    /*
    let result_pixels = block_match(&left_pixels, &right_pixels, width, height, block_w, block_h, diff_len);
    let result_pixels = normalize_result_pixels(&result_pixels, diff_len);
    let mut pixels = vec![];
    for p in result_pixels {
        pixels.extend(hsv_to_rgb(p as u8, 255, 255));
    }
    let result_image = RgbImage::from_raw(width as u32, height as u32, pixels).unwrap();
    let _saved = result_image.save("result.png");
    */

    let src = r#"
        __kernel void get_diffs(
                     __global unsigned char* left_pixels,
                     __global unsigned char* right_pixels,
                     __global unsigned char* diffs,
                     size_t w,
                     size_t h,
                     size_t diff_len) {
            size_t x = get_global_id(0);
            size_t y = get_global_id(1);
            size_t diff_index = get_global_id(2);
            size_t target_index = y * w + x;
            unsigned char left = left_pixels[target_index + diff_index];
            unsigned char right = right_pixels[target_index];
            unsigned char value;
            if (left > right)
                value = left - right;
            else
                value = right - left;
            diffs[target_index * diff_len + diff_index] = value;
        }

        __kernel void get_result_diffs(
                     __global unsigned char* diffs,
                     __global unsigned char* result_diffs,
                     size_t w,
                     size_t h,
                     size_t block_w,
                     size_t block_h,
                     size_t result_w,
                     size_t result_h,
                     size_t diff_len) {
            size_t result_x = get_global_id(0);
            size_t result_y = get_global_id(1);
            size_t z = get_global_id(2);
            if (result_x > result_w || result_y > result_h || z != 0)
                return;
            size_t x, y, i;
            size_t min_diff_index;
            unsigned int min_diff_point;
            for (i = 0; i < diff_len; i++) {
                unsigned int diff_point = 0;
                for (x = result_x * block_w; x < (result_x + 1) * block_w; x++) {
                    for (y = result_y * block_h; y < (result_y + 1) * block_h; y++) {
                        diff_point += (unsigned int) diffs[(y * w + x) * diff_len + i];
                    }
                }
                if (i == 0 || min_diff_point > diff_point) {
                    min_diff_index = i;
                    min_diff_point = diff_point;
                }
            }
            result_diffs[result_y * result_w + result_x] = min_diff_index;
        }
    "#;

    let global_work_size = SpatialDims::new(Some(width),Some(height),Some(diff_len)).unwrap();
    let pro_que = ProQue::builder()
        .src(src)
        .dims(global_work_size)
        .build().expect("Build ProQue");

    let left_pixels_buffer = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write().copy_host_ptr())
        .len(width * height)
        .host_data(&left_pixels)
        .build().unwrap();

    let right_pixels_buffer = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write().copy_host_ptr())
        .len(width * height)
        .host_data(&right_pixels)
        .build().unwrap();

    let diffs_buffer: Buffer<u8> = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(width * height * diff_len)
        .build().unwrap();

    let result_w = width / block_w;
    let result_h = height/ block_h;

    let result_diffs_buffer: Buffer<u8> = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(result_w * result_h)
        .build().unwrap();

    let result_pixels_buffer: Buffer<u8> = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(result_w * result_h)
        .build().unwrap();

    let get_diffs_kernel = pro_que.create_kernel("get_diffs").unwrap()
        .arg_buf(&left_pixels_buffer)
        .arg_buf(&right_pixels_buffer)
        .arg_buf(&diffs_buffer)
        .arg_scl(width)
        .arg_scl(height)
        .arg_scl(diff_len);

    unsafe { get_diffs_kernel.enq().unwrap(); }

    let get_result_diffs_kernel = pro_que.create_kernel("get_result_diffs").unwrap()
        .arg_buf(&diffs_buffer)
        .arg_buf(&result_diffs_buffer)
        .arg_scl(width)
        .arg_scl(height)
        .arg_scl(block_w)
        .arg_scl(block_h)
        .arg_scl(result_w)
        .arg_scl(result_h)
        .arg_scl(diff_len);

    unsafe { get_result_diffs_kernel.enq().unwrap(); }

    let mut result_diffs = vec![0; result_diffs_buffer.len()];
    result_diffs_buffer.read(&mut result_diffs).enq().unwrap();

    //println!("{:?}", result_diffs);
    let mut pixels = vec![];
    let diff_len_f32 = diff_len as f32;
    for p in result_diffs {
        let h = ((diff_len_f32 - p as f32) / diff_len_f32) * 200.0;
        pixels.extend(hsv_to_rgb(h as u8, 255, 255));
    }
    let result_image = RgbImage::from_raw(result_w as u32, result_h as u32, pixels).unwrap();
    let _saved = result_image.save("result.png");
}
