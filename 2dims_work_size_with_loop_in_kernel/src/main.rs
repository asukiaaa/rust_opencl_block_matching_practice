extern crate image;
extern crate ocl;
extern crate time;
use image::{GenericImage, RgbImage};
use ocl::{Buffer, MemFlags, ProQue, SpatialDims};
use time::PreciseTime;

fn get_gray_pixels(file_name: &str) -> (Vec<u8>, usize, usize) {
    let img = image::open(file_name).unwrap().grayscale();
    (img.raw_pixels(), img.width() as usize, img.height() as usize)
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
    let start_time = PreciseTime::now();
    // let left_image_file_name = "../data/aloeL.jpg";
    // let right_image_file_name = "../data/aloeR.jpg";
    let left_image_file_name = "../data/left.png";
    let right_image_file_name = "../data/right.png";
    let (left_pixels, width, height) = get_gray_pixels(&left_image_file_name);
    let (right_pixels, _, _) = get_gray_pixels(&right_image_file_name);
    let block_w = 11;
    let block_h = 11;
    let diff_len = width / 4;

    let loaded_image_time = PreciseTime::now();

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
            size_t target_index = y * w + x;
            size_t diff_index;
            for (diff_index = 0; diff_index < diff_len; ++diff_index) {
                unsigned char left = left_pixels[target_index + diff_index];
                unsigned char right = right_pixels[target_index];
                unsigned char value;
                if (left > right)
                    value = left - right;
                else
                    value = right - left;
                diffs[target_index * diff_len + diff_index] = value;
            }
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
            if (result_x > result_w || result_y > result_h)
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

    let global_work_size = SpatialDims::new(Some(width),Some(height),Some(1)).unwrap();
    let pro_que = ProQue::builder()
        .src(src)
        .dims(global_work_size)
        .build().expect("Build ProQue");

    let put_kernel_time = PreciseTime::now();

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

    let create_buffer_time = PreciseTime::now();

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

    let got_result_time = PreciseTime::now();

    //println!("{:?}", result_diffs);
    let mut pixels = vec![];
    let diff_len_f32 = diff_len as f32;
    for p in result_diffs {
        let h = ((diff_len_f32 - p as f32) / diff_len_f32) * 200.0;
        pixels.extend(hsv_to_rgb(h as u8, 255, 255));
    }
    let result_image = RgbImage::from_raw(result_w as u32, result_h as u32, pixels).unwrap();
    let _saved = result_image.save("result.png");

    let created_result_image_time = PreciseTime::now();

    println!("Load image {} sec", start_time.to(loaded_image_time));
    println!("Put kernel {} sec", loaded_image_time.to(put_kernel_time));
    println!("Create buffer {} sec", put_kernel_time.to(create_buffer_time));
    println!("Get result {} sec", create_buffer_time.to(got_result_time));
    println!("Create resutl image {} sec", got_result_time.to(created_result_image_time));
    println!("Total {} sec", start_time.to(created_result_image_time));
}
