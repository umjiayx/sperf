clear
clc

patient_ids_all = {'patient1_c1_s2', 'patient2_c1_s2', 'patient3_c1_s2', ...
               'patient4_c1_s2', 'patient5_c1_s2', 'patient6_c1_s2'};


for idx = 1:1:size(patient_ids_all, 2)
    PSMA_patient_id = patient_ids_all{idx};

    psma_full_combine_two_beds(PSMA_patient_id);
    psma_partial_combine_two_beds(PSMA_patient_id);
    psma_linear_combine_two_beds(PSMA_patient_id);
    psma_sperf_combine_two_beds(PSMA_patient_id);
end