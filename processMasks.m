function [allMasks, boundingBoxes] = processMasks(maskPaths)
    allMasks = [];
    boundingBoxes = [];

    for i = 1:numel(maskPaths)
        maskImage = imread(maskPaths{i});
        maskImage = imbinarize(maskImage);

        labeledMask = bwlabel(maskImage);
        stats = regionprops(labeledMask, 'BoundingBox');

        numInstances = length(stats);
        masks3D = false(size(maskImage, 1), size(maskImage, 2), numInstances);
        boxes = zeros(numInstances, 4);

        for k = 1:numInstances
            currentMask = labeledMask == k;
            masks3D(:, :, k) = currentMask;
            boxes(k, :) = stats(k).BoundingBox;
        end

        allMasks = cat(3, allMasks, masks3D);
        boundingBoxes = [boundingBoxes; boxes];
    end
end