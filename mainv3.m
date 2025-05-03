clc;
clear all;
close all;


%% Extract Folders


baseImageDir = 'C:\MIASVI\dataSet\imagesDataSet';  %%TUTAJ ZMIEN
baseMaskDir = 'C:\MIASVI\dataSet\masks';   %%TUTAJ ZMIEN

imageSubfolders = dir(baseImageDir);
maskSubfolders = dir(baseMaskDir);

% Skipping bad folders
imageSubfolders = imageSubfolders([imageSubfolders.isdir] & ~ismember({imageSubfolders.name}, {'.', '..'}));
maskSubfolders = maskSubfolders([maskSubfolders.isdir] & ~ismember({maskSubfolders.name}, {'.', '..'}));

extractName = @(s) strrep(strrep(s, 'images ', ''), 'masks ', '');
imageFolderNames = cellfun(extractName, {imageSubfolders.name}, 'UniformOutput', false);
maskFolderNames = cellfun(extractName, {maskSubfolders.name}, 'UniformOutput', false);

commonFolders = intersect(imageFolderNames, maskFolderNames);


%% Altering through folders


allMatchedImagePaths = {};
allMatchedMaskPaths = {};

for i = 1:length(commonFolders)
    datasetName = commonFolders{i};

    imageFolder = imageSubfolders(strcmp(imageFolderNames, datasetName)).name;
    maskFolder = maskSubfolders(strcmp(maskFolderNames, datasetName)).name;
    
    imageDir = fullfile(baseImageDir, imageFolder);
    maskDir = fullfile(baseMaskDir, maskFolder);
    
    fprintf('Processing dataset: %s (Images: %s, Masks: %s)\n', datasetName, imageFolder, maskFolder);

    % TIF and PNG
    imageFiles = [dir(fullfile(imageDir, '*.tif')); dir(fullfile(imageDir, '*.png'))];
    maskFiles = [dir(fullfile(maskDir, '*.tif')); dir(fullfile(maskDir, '*.png'))];

    %GTFO if empty
    if isempty(imageFiles) || isempty(maskFiles)
        fprintf('Skipping dataset "%s" (no images or masks found)\n', datasetName);
        continue;
    end

    % normalize names
    imageNames = lower(strtrim(erase({imageFiles.name}, {'.tif', '.png'})));
    maskNames = lower(strtrim(erase({maskFiles.name}, {'.tif', '.png'})));

    if isempty(maskNames)
        fprintf('No masks found in "%s", skipping.\n', datasetName);
        continue;
    end

    maskSet = containers.Map(maskNames, 1:length(maskNames));

    matchedImagePaths = {};
    matchedMaskPaths = {};

    for j = 1:length(imageNames)
        if isKey(maskSet, imageNames{j}) 
            matchedImagePaths{end+1} = fullfile(imageFiles(j).folder, imageFiles(j).name);
            matchedMaskPaths{end+1} = fullfile(maskFiles(maskSet(imageNames{j})).folder, maskFiles(maskSet(imageNames{j})).name);
        end
    end

    % variable for all the pairs
    allMatchedImagePaths = [allMatchedImagePaths, matchedImagePaths];
    allMatchedMaskPaths = [allMatchedMaskPaths, matchedMaskPaths];

    % debug shit
    fprintf('Matched %d pairs in %s\n', length(matchedImagePaths), datasetName);
end


%% Create datastores (test)


className = {['Adipocyte']};
startIndexBatch = 908;
endIndexBatch = 958; 
dataDir = 'C:\MIASVI\dataSet';              %%TUTAJ ZMIEN
annsDir = fullfile(dataDir, 'annotations'); %%W FOLDERZE Z GORY DODAJ FOLDER 'annotations'

createDataSet = false;

if createDataSet
    for i = startIndexBatch:endIndexBatch
        img = imread(allMatchedImagePaths{i});
        
        [logicalMasks, boundingBoxes] = processMasks({allMatchedMaskPaths{i}});
    
        slabels = categorical(repmat({'Adipocyte'}, size(boundingBoxes, 1), 1), className);
    
        imageFile = allMatchedImagePaths{i}; 
        [~, imageName, ~] = fileparts(imageFile); 
        savePath = fullfile(dataDir, 'annotations', [imageName, '.mat']);
    
        save(savePath, 'imageFile', 'boundingBoxes', 'slabels', 'logicalMasks', '-v7.3');
    end
end

ds = fileDatastore(annsDir,FileExtensions=".mat",ReadFcn=@(x)matReader(x,dataDir));

numImages = length(ds.Files);
numTrain = floor(0.8*numImages);
numVal = floor(0.1*numImages);

shuffledIndices = randperm(numImages);
trainDS = subset(ds,shuffledIndices(1:numTrain));
valDS   = subset(ds,shuffledIndices(numTrain+1:numTrain+numVal));
testDS  = subset(ds,shuffledIndices(numTrain+numVal+1:end));

%% Preview

sSample = preview(trainDS);
sImg = sSample{1};
sboxes = sSample{2};
slabels = sSample{3};
smasks  = sSample{4};

overlayedMasks = insertObjectMask(sImg,smasks,Opacity=0.5);
imshow(overlayedMasks)
showShape("rectangle",sboxes,Label=string(slabels),Color="green");


%% Create solov2 and its options


options = trainingOptions("sgdm", ...
    InitialLearnRate=0.0005, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=1, ...
    LearnRateDropFactor=0.99, ...
    Momentum=0.9, ...
    MaxEpochs=5, ...
    MiniBatchSize=4, ...
    ExecutionEnvironment="auto", ...
    VerboseFrequency=5, ...
    Plots="training-progress", ...
    ResetInputNormalization=false, ...
    ValidationData=valDS, ...
    ValidationFrequency=10, ... 
    GradientThreshold=35, ...
    OutputNetwork="best-validation-loss");


baseModel = solov2("resnet50-coco", className, InputSize=[1024 1024 3]);


%% Train the beast (its stupid for now)


doTraining = true;
if doTraining       
    net = trainSOLOV2(trainDS,baseModel,options,FreezeSubNetwork="backbone");
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save(fullfile(tempdir,"trainedSOLOv2"+modelDateTime+".mat"), ...
        "net");
else
    load("trainedSOLOv2.mat");
end

save('traineModel', 'net', '-v7.3');

%% Evaluate 
load('trainedModelMetrics.mat');
load('traineModel.mat');


folderPath = 'SegmentObjectResults';  

if exist(folderPath, 'dir') 
    rmdir(folderPath, 's');  
end

resultsDS = segmentObjects(net,testDS,Threshold=0.1);
%metrics = evaluateInstanceSegmentation(resultsDS, testDS, 0.5);
summarize(metrics)

display(metrics.ImageMetrics)

[precision,recall] = precisionRecall(metrics);
averagePrecision = averagePrecision(metrics);

figure
plot(recall{:},precision{:})
title(sprintf("Average Precision for Single Class Instance Segmentation: " + "%.2f",averagePrecision))
xlabel("Recall")
ylabel("Precision")
grid on
save('trainedModelMetrics', 'metrics');

%% Show Overlayed Masks

reset(testDS);
reset(resultsDS);
while hasdata(testDS)
    data = read(testDS);     
    testImage = data{1};        
    segmentedData = read(resultsDS);

    masks = segmentedData{1};
    labels = segmentedData{3};
    maskColors = lines(numel(labels));
    boundingBox = processMasks(allMatchedMaskPaths(i));
    overlayedImage = insertObjectMask(testImage, masks, Color=maskColors);
    imshow(overlayedImage);
    boxes = processSegmentedMasks(masks);
    showShape("rectangle",boxes,Label=string(labels),Color="green");
    waitforbuttonpress;
end



