clc
close all
clear all

load 'Base_Layers' 
lgraph = removeLayers(lgraph,'inputImage'); 
inputImage = imageInputLayer([500 500 3],'Name','inputImage'); 
lgraph = addLayers(lgraph, inputImage);
lgraph = connectLayers(lgraph,'inputImage','conv1_1'); 
lgraph = removeLayers(lgraph,{'conv5_1','bn_conv5_1','relu5_1','conv5_2','bn_conv5_2','relu5_2','conv5_3','bn_conv5_3','relu5_3','pool5','decoder5_unpool','decoder5_conv3','decoder5_bn_3','decoder5_relu_3','decoder5_conv2','decoder5_bn_2','decoder5_relu_2','decoder5_conv1','decoder5_bn_1','decoder5_relu_1'});
lgraph = connectLayers(lgraph,'pool4/out','decoder4_unpool/in');

lgraph = disconnectLayers(lgraph,'decoder1_relu_2','decoder1_conv1');
d_add1=additionLayer(2,'Name','d_add1');
lgraph = addLayers(lgraph, d_add1);
lgraph = connectLayers(lgraph,'decoder1_relu_2','d_add1/in1'); 
lgraph = connectLayers(lgraph,'relu1_1','d_add1/in2');
lgraph = connectLayers(lgraph,'d_add1','decoder1_conv1'); 

lgraph = disconnectLayers(lgraph,'decoder2_relu_2','decoder2_conv1');
d_add2=additionLayer(2,'Name','d_add2');
lgraph = addLayers(lgraph, d_add2);
lgraph = connectLayers(lgraph,'decoder2_relu_2','d_add2/in1'); 
lgraph = connectLayers(lgraph,'relu2_1','d_add2/in2');
lgraph = connectLayers(lgraph,'d_add2','decoder2_conv1'); 

lgraph = disconnectLayers(lgraph,'decoder3_relu_2','decoder3_conv1');
d_add3=additionLayer(2,'Name','d_add3');
lgraph = addLayers(lgraph, d_add3);
lgraph = connectLayers(lgraph,'decoder3_relu_2','d_add3/in1'); 
lgraph = connectLayers(lgraph,'relu3_1','d_add3/in2');
lgraph = connectLayers(lgraph,'d_add3','decoder3_conv1'); 

lgraph = disconnectLayers(lgraph,'decoder4_relu_2','decoder4_conv1');
d_add4=additionLayer(2,'Name','d_add4');
lgraph = addLayers(lgraph, d_add4);
lgraph = connectLayers(lgraph,'decoder4_relu_2','d_add4/in1'); 
lgraph = connectLayers(lgraph,'relu4_1','d_add4/in2');
lgraph = connectLayers(lgraph,'d_add4','decoder4_conv1'); 

Folder = '';%directory to all images
train_img_dir = fullfile(Folder,'Train folder name');
imds = imageDatastore(train_img_dir); 
 
classes = ["nuclei","background"];
labelIDs   = [255 0];

train_label_dir = fullfile(Folder,'Train GT');
pxds = pixelLabelDatastore(train_label_dir,classes,labelIDs);

% Class balancing
tbl = countEachLabel(pxds);
frequency = tbl.PixelCount/sum(tbl.PixelCount);
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq; 

pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights); % adding weights tp pixel classification layer

lgraph = removeLayers(lgraph,'pixelLabels');
lgraph = addLayers(lgraph, pxLayer); 
lgraph = connectLayers(lgraph,'softmax','labels');

options = trainingOptions('adam', ...
    'SquaredGradientDecayFactor',0.95, ...
    'GradientThreshold',8, ...
    'GradientThresholdMethod','global-l2norm', ...
    'Epsilon',1e-6, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.0005, ...
    'MaxEpochs',30, ...  
    'MiniBatchSize',4, ...
    'CheckpointPath',tempdir, ...
    'Shuffle','every-epoch', ...
    'VerboseFrequency',2);
augment_data = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-5 5],'RandYTranslation',[-5 5]); % optional data augmentation
training_data = pixelLabelImageDatastore(imds,pxds,...
    'DataAugmentation',augment_data); %% complete image+label data

[net, info] = trainNetwork(training_data,lgraph,options);% Train the network
Trained_net=net;
save Trained_net;
