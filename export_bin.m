%% setup location of .bin files

path = '~/projects/nmf/';


%% write test matrices to binary files

X = [1 2 3; 4 5 6; 7 8 9];
W = [1 2; 3 4; 5 6; 7 8];
H = [1 2 3; 4 5 6]; 



fid = fopen([path 'Xs.bin'],'w');


fwrite(fid,size(X),'int');
count = fwrite(fid,X(:),'float');
fprintf('wrote file with %u elements\n',count)

fclose(fid);



fid = fopen([path 'Ws.bin'],'w');

fwrite(fid,size(W),'int');
count = fwrite(fid,W(:),'float');
fprintf('wrote file with %u elements\n',count)

fclose(fid);

fid = fopen([path 'Hs.bin'],'w');

fwrite(fid,size(H),'int');
count = fwrite(fid,H(:),'float');
fprintf('wrote file with %u elements\n',count)

fclose(fid);



%% write out binary matrices stored in X,W,H with dimensions as first two ints

fid = fopen([path 'X2.bin'],'w');


fwrite(fid,size(X),'int');
count = fwrite(fid,X(:),'float');
fprintf('wrote file with %u elements\n',count)

fclose(fid);



fid = fopen([path 'W2.bin'],'w');

fwrite(fid,size(W),'int');
count = fwrite(fid,W(:),'float');
fprintf('wrote file with %u elements\n',count)

fclose(fid);

fid = fopen([path 'H2.bin'],'w');

fwrite(fid,size(H),'int');
count = fwrite(fid,H(:),'float');
fprintf('wrote file with %u elements\n',count)

fclose(fid);



%% read in binary matrix Wout.bin and Hout.bin

fid = fopen([path 'hmat.bin'],'r');


dim = fread(fid,2,'int');
mat = fread(fid,dim(1)*dim(2),'float');
Wout = reshape(mat,dim(1),dim(2));


fclose(fid);


%%



fid = fopen([path 'Hout.bin'],'r');


dim = fread(fid,2,'int');
mat = fread(fid,dim(1)*dim(2),'float');
Hout = reshape(mat,dim(1),dim(2));


fclose(fid);



% order components by energy
[Wout,Hout,en]=order_comp(Wout,Hout);
% normalize rows of Hout to [0,1]
Hmax = repmat(max(Hout,[],2)',size(Wout,1),1);
Wout = Wout .* Hmax;
Hmax = repmat(max(Hout,[],2),1,size(Hout,2));
Hout = Hout ./ Hmax;


subplot(2,1,1)
imagesc(20*log10(Wout),[0 100])
set(gca,'ydir','normal');

subplot(2,1,2)
imagesc(Hout);





