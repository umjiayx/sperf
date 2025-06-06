clear
clc

patient_ids_all = {'patient1_c1_s2', 'patient2_c1_s4', 'patient3_c2_s4', ...
    'patient4_c1_s4', 'patient5_c1_s4', 'patient6_c1_s4', 'patient7_c1_s4', ...
    'patient8_c1_s4', 'patient9_c2_s2', 'patient10_c1_s4', 'patient11_c1_s4'};

%% Params
df = 2;
sl_st = 24; 
sl_end = 104;
downfactor = df;
if df == 2
    start = 2;
else
    start = 1;
end

NRMSE_list_linear = zeros(1, length(patient_ids_all));
NRMSE_list_nerf = zeros(1, length(patient_ids_all));

for id =1:1:size(patient_ids_all, 2)
    DOTATATE_patient_id = patient_ids_all{id};
    proj_path = strcat('./proj/proj_', DOTATATE_patient_id, '.fld');
    proj = fld_read(proj_path);
    proj = proj(:,sl_st:sl_end,:); % 83 slices

    %% Linear
    yi = proj(:,:,1:120); % 208 main window
    yi_tot = zeros(size(yi));
    yi = yi(:,:,start:downfactor:end);
    yi_tot(:,:,start:downfactor:end) = yi;
    
    test_view_index = true(1,120);
    test_view_index(start:downfactor:end) = false;

    % Check downfactor == 2,4,8
    if downfactor == 2 || downfactor == 4 || downfactor == 8
        for j = 2:1:downfactor
            for i = j:downfactor:size(yi_tot, 3)
               start_idx = (i+downfactor-j)/downfactor;
               end_idx = start_idx + 1;
               if end_idx > (size(yi_tot, 3) / downfactor)
                   end_idx = 1;
               end
               coeff = (downfactor-j+1)/downfactor;
               if downfactor == 2
                   idx = mod(i + 1, size(yi_tot, 3));
               else
                   idx = i;
               end
               yi_tot(:,:,idx) = coeff * yi(:,:,start_idx) + (1-coeff) * yi(:,:,end_idx);
            end
        end
    else
        error("Invalid downfactor!!! Please double check!!! \n");
    end
    
    yi_l = yi_tot;
    NRMSE_linear = cal_NRMSE(yi_l(:,:,test_view_index), proj(:,:,test_view_index));
    NRMSE_list_linear(id) = NRMSE_linear;

    %% NeRF
    pred_proj_path = strcat('./proj_sperf_mat/', DOTATATE_patient_id, '/test/pred_proj_test.mat');
    gt_proj_path = strcat('./proj_sperf_mat/', DOTATATE_patient_id, '/test/gt_proj_test.mat');
    results = load(pred_proj_path).results;
    GT = load(gt_proj_path).GT;
    pred = imresize(results, 0.5);
    gt = imresize(GT, 0.5);
    NRMSE_nerf = cal_NRMSE(pred, gt);
    NRMSE_list_nerf(id) = NRMSE_nerf;
end

disp(mean(NRMSE_list_linear))
disp(mean(NRMSE_list_nerf))

