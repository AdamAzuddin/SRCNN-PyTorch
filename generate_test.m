clear; close all;

%% Settings
folder = 'Test/Set5';
savepath = 'test.h5';
size_input = 33;
size_label = 21;
scale = 3;
stride = 21;

%% Initialization
data = zeros(size_input, size_input, 3, 1);  
label = zeros(size_label, size_label, 3, 1);
padding = abs(size_input - size_label) / 2;
count = 0;

%% Generate data
filepaths = dir(fullfile(folder, '*.bmp'));

for i = 1 : length(filepaths)
    % Read image
    image = imread(fullfile(folder, filepaths(i).name));
    image = im2double(image);
    
    % Process image for label and input
    im_label = modcrop(image, scale);
    [hei, wid, ~] = size(im_label);  % Get image dimensions
    
    % Resize label image for input
    im_input = imresize(imresize(im_label, 1/scale, 'bicubic'), [hei, wid], 'bicubic');

    % Extract patches
    for x = 1 : stride : hei - size_input + 1
        for y = 1 : stride : wid - size_input + 1
            % Extract input and label patches
            subim_input = im_input(x : x + size_input - 1, y : y + size_input - 1, :);  % Include all RGB channels
            subim_label = im_label(x + padding : x + padding + size_label - 1, y + padding : y + padding + size_label - 1, :);  % Include all RGB channels

            % Increment count
            count = count + 1;
            
            % Assign patches to data and label arrays
            data(:, :, :, count) = subim_input;
            label(:, :, :, count) = subim_label;
        end
    end
end

%% Shuffle data
order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order);

%% Writing to HDF5
chunksz = 128;
created_flag = false;
totalct = 0;

for batchno = 1 : floor(count / chunksz)
    last_read = (batchno - 1) * chunksz + 1;
    batchdata = data(:, :, :, last_read : last_read + chunksz - 1);
    batchlabs = label(:, :, :, last_read : last_read + chunksz - 1);

    startloc = struct('dat', [1, 1, 1, totalct + 1], 'lab', [1, 1, 1, totalct + 1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end

%% Display HDF5 file information
h5disp(savepath);