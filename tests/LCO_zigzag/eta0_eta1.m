
clear
for i=5
    var = ['surrogate_model/predicting_' num2str(i)   '.txt'];
    % var = ['surrogate_model/training/hp_prediction_' num2str(i)   '.txt'];
    var = ['../../Output_zigzag/300/sim2/graphs/rnd' num2str(i)   '.txt'];
    data= load(var);
    % training = load('Output_2d/280_6/results0.txt');

    % data= data(:,1:end-1)

    figure(2*i+1);clf
    g = (reshape(data(:,end-1),[50,50])-min(data(:,end-1)));
    % g= (reshape(data(:,end),[50,50])-min(data(:,end)));
    surf(reshape(data(:,1),[50,50]),reshape(data(:,2),[50,50]),g,min(g,.003))
    % eta0=data(:,1);
    % eta1=data(:,2);
    % scatter3(eta0,eta1,(data(:,end)-min(data(:,end)))/100)
    % 
    % 
    % surf(eta0_1,eta1_1,free_1);
    % xlim([0.4,.6])
    % xlabel('\eta_0')
    % ylabel('\eta_1')
    % zlabel('g')
    % title('Free energy as a function of composition and order parameters')
    xlim([0.45,0.55])
    % % ylim([0,0.5])
    zlim([0 .003])
    % view(-65,16)
    % set(gca,'FontSize',16)
    % set(gca,'ydir','reverse')
    % shading interp 
    % % axis tight
    % grid on 
    % 
    % 
    % figure(2*i+2);clf
    % scatter(data(:,1),data(:,8))
    % xlim([0.45,0.55])
    % 
    % var2 = ['results.txt'];
    % data2 = load(var2);
    % 
    % figure(2*i+3);clf
    % scatter(data(:,1),data(:,8))
    % hold on 
    % scatter(data2(:,8),data2(:,23))
    % ylim([-0.8 0.8])
    % 
    % 
    % figure(i+2);clf
    % % g = (reshape(data(:,end),[50,50])-min(data(:,end)))/100;
    % % surf(reshape(data(:,1),[50,50]),reshape(data(:,2),[50,50]),g,min(g,.25))
    % eta0=data(:,1);
    % eta1=data(:,2);
    % % scatter3(eta0,eta1,(data(:,end)-min(data(:,end)))/100)
    % hold on
    % scatter3(training(:,1),training(:,2),training(:,end))
    % % xlim([0.495,0.595])
    % 
    % 
    % % surf(eta0_1,eta1_1,free_1);
    % % xlim([0.4,.6])
    % xlabel('\eta_0')
    % ylabel('\eta_1')
    % zlabel('formation energy')
    % % title('0 K')
    % xlim([0.45,0.55])
    % ylim([0,0.5])
    % zlim([-.168 -.158])
    % view(-65,16)
    % set(gca,'FontSize',16)
    % set(gca,'ydir','reverse')
    % shading interp 
    % grid on 
    % hold off
    % % savefig('graph.fig')
    % 
    % figure(i+1);clf
    % mu0=data(:,3);
    % eta0=data(:,1);
    % eta1=data(:,2);
    % scatter3(eta0,mu0,eta1)
    % hold on
    % % scatter3(training(:,1),training(:,6),training(:,2))
    % legend('prediction','training')
    % xlabel('\eta_0')
    % ylabel('\mu_0')
    % zlabel('\eta_1')
    % xlim([0.45,0.55])
    % % xlim([0.48,0.52])
    % view(-65,16)
end
% 