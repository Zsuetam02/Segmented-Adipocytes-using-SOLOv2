function boundingBoxes = processSegmentedMasks(masks)
    numInstances = size(masks, 3);
    boundingBoxes = zeros(numInstances, 4); 

    for i = 1:numInstances
        singleMask = masks(:, :, i); 
        stats = regionprops(singleMask, 'BoundingBox');
        
        if ~isempty(stats)
            boundingBoxes(i, :) = stats(1).BoundingBox;
        end
    end
end