#[macro_export]
macro_rules! init_array {
    ([$value_type:ty; $count:expr], mut $value_fn:expr) => {{
        let mut func = $value_fn;
        let mut array: std::mem::MaybeUninit<[$value_type; $count]> =
            std::mem::MaybeUninit::uninit();
        unsafe {
            let mut idx = 0;
            while idx < $count {
                (array.as_mut_ptr() as *mut $value_type)
                    .add(idx)
                    .write(func(idx));
                idx += 1;
            }
            array.assume_init()
        }
    }};
    ([$value_type:ty; $count:expr], $value_fn:expr) => {{
        let func = $value_fn;
        let mut array: std::mem::MaybeUninit<[$value_type; $count]> =
            std::mem::MaybeUninit::uninit();
        unsafe {
            let mut idx = 0;
            while idx < $count {
                (array.as_mut_ptr() as *mut $value_type)
                    .add(idx)
                    .write(func(idx));
                idx += 1;
            }
            array.assume_init()
        }
    }};
    ([$value_type:ty; $count:expr], ($($passed:expr),*$(,)?), const mut $value_fn:expr) => {{
        let mut func = $value_fn;
        let mut array: std::mem::MaybeUninit<[$value_type; $count]> =
            std::mem::MaybeUninit::uninit();
        unsafe {
            let mut idx = 0;
            while idx < $count {
                (array.as_mut_ptr() as *mut $value_type)
                    .add(idx)
                    .write(func(idx,$($passed),*));
                idx += 1;
            }
            array.assume_init()
        }
    }};
    ([$value_type:ty; $count:expr], ($($passed:expr),*$(,)?), const $value_fn:expr) => {{
        let func = $value_fn;
        let mut array: std::mem::MaybeUninit<[$value_type; $count]> =
            std::mem::MaybeUninit::uninit();
        unsafe {
            let mut idx = 0;
            while idx < $count {
                (array.as_mut_ptr() as *mut $value_type)
                    .add(idx)
                    .write(func(idx,$($passed),*));
                idx += 1;
            }
            array.assume_init()
        }
    }};
}
