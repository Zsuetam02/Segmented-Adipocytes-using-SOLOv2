%%%%%%REWORK DATASET%%%%%%%%

clc;
clear all;

dir1 = 'C:\MIASVI\dataSet\imagesDataSet'; 
dir2 = 'C:\MIASVI\dataSet\masks'; 

imageFiles1 = dir(fullfile(dir1, '**', '*.*'));
imageFiles2 = dir(fullfile(dir2, '**', '*.*')); 

imageFiles1 = imageFiles1(~[imageFiles1.isdir] & (endsWith({imageFiles1.name}, {'.png', '.tiff', '.tif'})));
imageFiles2 = imageFiles2(~[imageFiles2.isdir] & (endsWith({imageFiles2.name}, {'.png', '.tiff', '.tif'})));

allImageFiles = [imageFiles1; imageFiles2];

for i = 1:length(allImageFiles)
    currentImagePath = fullfile(allImageFiles(i).folder, allImageFiles(i).name);
    img = imread(currentImagePath);

    [height, width, ~] = size(img);
    if height ~= 1024 || width ~= 1024
        resizedImg = imresize(img, [1024 1024]);

        imwrite(resizedImg, currentImagePath);
        disp(['Resized image: ', currentImagePath]);
    else
        disp(['Image already 1024x1024: ', currentImagePath]);
    end
end