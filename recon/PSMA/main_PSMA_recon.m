clear
clc

patient_ids_all = {'02sp_c1_s2', '03nm_c1_s2', '05rd_c1_s2', ...
               '07ga_c1_s2', '08os_c1_s2', '09ld_c1_s2', }; ...
               % 'cl2_jf_c1_s2', 'cl5_ck_c1_s2', 'cl9_fb_c1_s2', 'cl11_rr_c1_s2'};


for idx = 1:1:size(patient_ids_all, 2)
    PSMA_patient_id = patient_ids_all{idx};

    psma_full_combine_two_beds(PSMA_patient_id);
    psma_partial_combine_two_beds(PSMA_patient_id);
    psma_linear_combine_two_beds(PSMA_patient_id);
    psma_sperf_combine_two_beds(PSMA_patient_id);
end