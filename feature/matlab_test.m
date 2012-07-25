
im = imread('..\test\test.jpg');

norient = 18;
sbin = 4;
pixel_opt.name = 'HOG';
pixel_opt.image_depth = 1;
pixel_opt.param = norient;
pixel_opt.length = norient;
pixel_opt.length_sparse = 1;

opt.pixel_opt = pixel_opt;
opt.numbin_x = 1;
opt.numbin_y = 1;
opt.sizebin_x = sbin;
opt.sizebin_y = sbin;
opt.length = pixel_opt.length;
im = rgb2gray(im);
% [feat, coordinate] = patch_feature(rgb2gray(im), opt);
% HOG  0.0282s vs UoC HOG  0.0231s
tic;
for i = 1:1000
    feat = patch_feature(im, opt);
end
disp(toc/1000);
% feat = feat(:,:,1:8) + feat(:,:,9:16);
% feat = bsxfun(@rdivide, feat, sqrt(sum(feat.^2, 3)));
% imshow(FeatureVisualizeDenseHOG(feat, [], 20));