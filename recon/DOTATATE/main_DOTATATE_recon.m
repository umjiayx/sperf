patient_ids_all = {'patient1_c1_s2', 'patient2_c1_s4', 'patient3_c2_s4', ...
    'patient4_c1_s4', 'patient5_c1_s4', 'patient6_c1_s4', 'patient7_c1_s4', ...
    'patient8_c1_s4', 'patient9_c2_s2', 'patient10_c1_s4', 'patient11_c1_s4'};


for idx =1:1:size(patient_ids_all, 2)
    DOTATATE_patient_id = patient_ids_all{idx};

    dotatate_full_recon(DOTATATE_patient_id);
    dotatate_partial_recon(DOTATATE_patient_id);
    dotatate_linear_recon(DOTATATE_patient_id);
    dotatate_sperf_recon(DOTATATE_patient_id);
    dotatate_BerThin_recon(DOTATATE_patient_id);
end

