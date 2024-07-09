clear

i=2;
var = ['../../Output/260_5/data/CASMallresults_extended' num2str(i)   '_no_header.txt'];
data= load(var);

eta_0 = data(:,5);
eta_1 = max(abs(data(:,6:8)),[],2);
eta_2 = data(:,7);
eta_3 = data(:,8);
mu_0 = data(:,14);
% potential = data(:,18)

figure(1);clf

scatter(eta_0,mu_0)
hold on

% figure(2);clf
% 
% scatter3(eta_0,eta_1,potential)
% xlim([0.45,0.55])
% zlim([-.165 -.158])
% hold on