target_dir = '/usr/project/xtmp/DIV2K_train_HR';
count = 0;
factor = 8;
patch_size = 400;
f_lst = [];
f_lst = [f_lst; dir(fullfile(target_dir, '*.png'))];
for f = 1:numel(f_lst)
    %fname = sprintf('%04d.png', i);
    file = f_lst(f);
    fname = file.name;
    if fname(1)=='.'
        continue;
    end
    target_path = fullfile(target_dir, fname);

    target_img = imread(target_path);
    
    img_size = size(target_img);

    target_img = target_img(1:img_size(1)- mod(img_size(1), factor), 1:img_size(2) - mod(img_size(2), factor), :);

    img_size = size(target_img);
    
    x_max = img_size(1) - patch_size;
    y_max = img_size(2) - patch_size;

    x = uint32(rand(1, 100) * x_max) + 1;
    y = uint32(rand(1, 100) * y_max) + 1;
    
    for i = 1:length(x)
        x_coord = x(i);
        y_coord = y(i);
    
        original_target_patch = target_img(x_coord:x_coord+patch_size-1, y_coord:y_coord+patch_size-1, :);
        original_input_patch = imresize(original_target_patch, 1/factor, 'bicubic');

        target_patch = original_target_patch;
        input_patch = original_input_patch;
        patch_name = sprintf('/usr/project/xtmp/EDSR_HR_train/%d', count);
        save(patch_name, 'target_patch', 'input_patch', '-v6');
        count = count + 1;
    
        target_patch = imrotate(original_target_patch, 90);
        input_patch = imrotate(original_input_patch, 90);
        patch_name = sprintf('/usr/project/xtmp/EDSR_HR_train/%d', count);
        save(patch_name, 'target_patch', 'input_patch', '-v6');
        count = count + 1;

        target_patch = imrotate(original_target_patch, 180);
        input_patch = imrotate(original_input_patch, 180);
        patch_name = sprintf('/usr/project/xtmp/EDSR_HR_train/%d', count);
        save(patch_name, 'target_patch', 'input_patch', '-v6');
        count = count + 1;

        target_patch = imrotate(original_target_patch, 270);
        input_patch = imrotate(original_input_patch, 270);
        patch_name = sprintf('/usr/project/xtmp/EDSR_HR_train/%d', count);
        save(patch_name, 'target_patch', 'input_patch', '-v6');
        count = count + 1;

        target_patch = flipdim(original_target_patch, 1);
        input_patch = flipdim(original_input_patch, 1);
        patch_name = sprintf('/usr/project/xtmp/EDSR_HR_train/%d', count);
        save(patch_name, 'target_patch', 'input_patch', '-v6');
        count = count + 1;

        target_patch = flipdim(original_target_patch, 2);
        input_patch = flipdim(original_input_patch, 2);
        patch_name = sprintf('/usr/project/xtmp/EDSR_HR_train/%d', count);
        save(patch_name, 'target_patch', 'input_patch', '-v6');
        count = count + 1;

        display(count);
    end
end
