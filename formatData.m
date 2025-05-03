function dataOut = formatData(image, mask)
    % Empty bounding boxes and labels (since masks are used)
    bboxes = {};  % Empty for instance segmentation
    labels = categorical();  % Empty categorical array
    dataOut = {image, bboxes, labels, mask};
end