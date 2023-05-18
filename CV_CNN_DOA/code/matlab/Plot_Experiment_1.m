% Plot the Spatial spectrum reconstruction results(i.e., EX1)
% Author: Shulin Hu
% Date: 15/05/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
f_reslut_root = '../../Result/data/EX1/';
%Select the results that require a plot:
% f_reslut = fullfile(f_reslut_root,'EX1_result_DOA_set_K_1.h5');
% f_reslut = fullfile(f_reslut_root,'EX1_result_DOA_set_K_2_lager.h5');
f_reslut = fullfile(f_reslut_root,'EX1_result_DOA_set_K_2_small.h5');
% f_reslut = fullfile(f_reslut_root,'EX1_result_DOA_set_K_3.h5');

spectrum = double(h5read(f_reslut, '/spectrum'));
GT_angles = double(h5read(f_reslut, '/GT_angles'));
figure();
plot([-60:1:60],spectrum,LineWidth=1)
hold on
plot(GT_angles,1,'.r',Markersize=20)
set(gca,'FontSize',14);
legend('Estimated spatial spectrum','True DOA','Location','southeast')
hold off
ylabel('Probability', 'interpreter','latex');
xlabel('Direction [$^\circ$]', 'interpreter','latex');
grid on;grid minor;
