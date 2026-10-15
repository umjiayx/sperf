function psma_full_combine_two_beds(PSMA_patient_id)
    patient_id = PSMA_patient_id; 
    
    %% Recon!
    psma_full_recon_bed1(patient_id);
    psma_full_recon_bed2(patient_id);
    
    %% Combine two beds
    bed1_folder = ['./recon/full_recon/psma_', patient_id, '_b1'];
    bed2_folder = ['./recon/full_recon/psma_', patient_id, '_b2'];
    full_bed1 = fld_read(fullfile(bed1_folder, 'fullrecon_79slices.fld'));
    full_bed2 = fld_read(fullfile(bed2_folder, 'fullrecon_79slices.fld'));
    % full_2beds = cat(3, full_bed1(:,:,1:79,:), full_bed2(:,:,4:82,:));
    full_2beds = cat(3, full_bed1(:,:,:,:), full_bed2(:,:,:,:));
    
    savefolder = ['./recon/full_recon/psma_', patient_id];
    if ~exist(savefolder, 'dir')
        mkdir(savefolder)
    end
    
    % 82-3 + 82-3
    %fld_write(fullfile(savefolder, 'fullrecon_isaac_20iters.fld'), full_2beds, 'type', 'xdr_float');
    %fld_write(fullfile(savefolder, 'fullrecon_isaac.fld'), full_2beds(:,:,:,10), 'type', 'xdr_float');
    
    fld_write(fullfile(savefolder, 'fullrecon_isaac_79+79_20iters.fld'), full_2beds, 'type', 'xdr_float');
    fld_write(fullfile(savefolder, 'fullrecon_isaac_79+79.fld'), full_2beds(:,:,:,10), 'type', 'xdr_float');

end
