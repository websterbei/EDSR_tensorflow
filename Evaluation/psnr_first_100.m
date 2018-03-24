originalDir = 'DIV2K_valid_HR';
outputDir = 'EDSR_9/output';
count = 0;
psnrv = [];
for i = 801:900
    fnameOriginal = strcat(sprintf('%04d', i), '.png');
    fnameOutput = strcat(sprintf('%04d', i), 'x8.png');
    f_path = fullfile(originalDir, fnameOriginal);
    img_original = imread(f_path);
    f_path = fullfile(outputDir, fnameOutput);
    img_output = imread(f_path);
    img_size = size(img_original);
    %img_output = imresize(img_output, [img_size(1), img_size(2)]);
    psnrv = [psnrv, NTIRE_PeakSNR_imgs(img_original, img_output, 8)];
end
display(psnrv)
display(mean(psnrv))
