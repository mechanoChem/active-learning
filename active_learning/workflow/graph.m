
dnn_260= load('output7/graphs/2d_rnd5.txt');

figure
g = reshape(dnn_260(:,end),[50,50])-min(dnn_260(:,end));
surf(reshape(dnn_260(:,1),[50,50]),reshape(dnn_260(:,2),[50,50]),g,min(g,5e-6))

xlabel('\eta_0')
ylabel('\eta_1')
zlabel('g')
title('260 K')
xlim([0.45,0.55])
zlim([0 1e-5])
view(-65,16)
set(gca,'FontSize',16)
set(gca,'ydir','reverse')
shading interp 
grid on 
hold off
savefig('graph.fig')
