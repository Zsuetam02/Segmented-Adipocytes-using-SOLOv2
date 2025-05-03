function out = matReader(filename, datasetRoot)
    data = load(filename);

    if isfile(data.imageFile)
        imagePath = data.imageFile; 
    else
        imagePath = fullfile(datasetRoot, 'imagesDataSet', data.imageFile);
    end

    if ~isfile(imagePath)
        error("Image file not found: %s", imagePath);
    end
    im = imread(imagePath);

    % DEBUGGING SHIT FOR BOUNDING BOXES
    if isfield(data, 'boundingBoxes')
        boxes = data.boundingBoxes;
    elseif isfield(data, 'boxes')
        boxes = data.boxes;
    else
        error("No boxes or boundingBoxes", filename);
    end

    
    masks = logical(data.logicalMasks);

    % OUTPUT
    numObjects = size(boxes, 1);
    out{1} = im;
    out{2} = data.boundingBoxes; 
    out{3} = repmat(categorical("Adipocyte"), [numObjects 1]); 
    out{4} = masks;
end

