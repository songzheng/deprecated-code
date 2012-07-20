
im = imread('..\test\test.jpg');

norient = 18;
sbin = 8;
pixel_opt.name = 'HOG';
pixel_opt.image_depth = 1;
pixel_opt.param = norient;
pixel_opt.length = norient;

opt.pixel_opt = pixel_opt;
opt.numbin_x = 1;
opt.numbin_y = 1;
opt.size_x = sbin;
opt.size_y = sbin;
opt.step_x = sbin;
opt.step_y = sbin;
opt.bin_length = pixel_opt.length;
opt.length = opt.bin_length * opt.numbin_x * opt.numbin_y;

%  0.0489s vs UoC HOG  0.0231s
tic;
for i = 1:1000
    feat = patch_feature(rgb2gray(im), opt);
end
disp(toc/1000);
feat = feat(:,:,1:8) + feat(:,:,9:16);
feat = bsxfun(@rdivide, feat, sqrt(sum(feat.^2, 3)));
imshow(FeatureVisualizeDenseHOG(feat, [], 20));