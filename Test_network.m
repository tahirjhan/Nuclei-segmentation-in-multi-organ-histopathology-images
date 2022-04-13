clc
close all
clear all
load 'TCGA_trained_model' % Trained model name

image=imread('sample_input.tif');
% Normalize Image
[Inorm H1 E1] = Strain_Normalization(image);
im=Inorm;
% Test
tic;
C = semanticseg(im,net);
k=C=='nuclei';
toc;
k1=k;
mask=double(k);
GT=imread('sample_GT.tif');  
GT=double(GT);
imR=im(:,:,1);
imG=im(:,:,2);
imB=im(:,:,3);
[x1,y1]=size(imR);
err=0;
for jj=1:x1
    for kk=1:y1
        if GT(jj,kk)==255 && mask(jj,kk)==1
             imR(jj,kk)=0;
              imG(jj,kk)=0;
               imB(jj,kk)=255;           
       elseif GT(jj,kk)==255 && mask(jj,kk)==0
           imR(jj,kk)=255;
           imG(jj,kk)=0;
           imB(jj,kk)=0;
           err=err+1;          
        elseif GT(jj,kk)==0 && mask(jj,kk)==1 

          imR(jj,kk)=0;
          imG(jj,kk)=255;
          imB(jj,kk)=0;
          err=err+1;       
        end
    end
end  
im1=cat(3,imR,imG,imB);
imshow(im1);


    




