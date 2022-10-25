%%creating image datastore
rootFolder=fullfile('database');
imds = imageDatastore(fullfile(rootFolder),'IncludeSubFolder',true,...
       'LabelSource','foldernames');

numClass=numel(categories(imds.Labels));

% splitting imds to training set and test set
[trainingSet,other]=splitEachLabel(imds,0.7,'randomize');
[validationSet,testSet]=splitEachLabel(other,0.5,'randomize');

%% using alexnet pretrained model and fine tuning
net=alexnet;
imageSize=net.Layers(1).InputSize;
layers=net.Layers;
layers(23)=fullyConnectedLayer(numClass);
layers(25)=classificationLayer;

%% Preporcessing image data
imageAugmenter = imageDataAugmenter('RandXReflection',true,'RandRotation',[-5 5]);

augmentedTrainingSet=augmentedImageDatastore(imageSize,trainingSet,...
    'ColorPreprocessing','gray2rgb','dataAugmentation',imageAugmenter);
augmentedValidationSet=augmentedImageDatastore(imageSize,validationSet,...
    'ColorPreprocessing','gray2rgb','dataAugmentation',imageAugmenter);
augmentedTestSet=augmentedImageDatastore(imageSize,testSet,...
    'ColorPreprocessing','gray2rgb');
%% defining training options and training 
opts=trainingOptions('sgdm',...
    'InitialLearnRate',0.001,...
    'ExecutionEnvironment','gpu',...
    'MaxEpochs',30,'MiniBatchSize',128,...
    'Shuffle','every-epoch', ...
    'ValidationData',augmentedValidationSet, ...
    'ValidationFrequency',20,...
    'ValidationPatience',3,...
    'Plots','training-progress');
aNet_crop=trainNetwork(augmentedTrainingSet,layers,opts);

%% Testing the test set accuracy by the use of trained classifier 
predictedLabels=classify(aNet_crop,augmentedTestSet);
accuracy=mean(predictedLabels==testSet.Labels);
X=['Accuracy is ',num2str(accuracy)];
disp(X)
figure;
plotconfusion(testSet.Labels,predictedLabels)
