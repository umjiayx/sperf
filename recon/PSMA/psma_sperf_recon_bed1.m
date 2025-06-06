function psma_sperf_recon_bed1(patient_id, patient_name, start)
    %% Parameters
    nx = 128;
    ny = 128;
    nz = 79;
    na = 120;
    niter = 20;
    num_block = 6;
    downfactor = 4;
    sl_st = 24;
    sl_end = 102;
    mu_st = 1;
    mu_end = 79;
    
    mumap_path = ['./mumap_fld/mumap_psma_test_', patient_name, '.fld'];
    psf_path = ['./psf/psf/psf_proj_psma_', patient_id, '_b1_208.fld'];
    proj1_path = ['./proj_fld/proj_psma_', patient_id, '_b1.fld'];
    foldername = strcat('./recon/sperf_recon/psma_', patient_id, '_b1/');
    filename = 'sperfrecon_79slices.fld';
    proj_nerf_path = strcat('./proj_sperf_mat/psma_', patient_id, '_b1/test/pred_proj_test.mat');
    scatter_nerf_path = strcat('./proj_sperf_mat/psma_', patient_id, '_b1_scatter/test/pred_proj_test.mat');

    if ~exist(foldername, 'dir')
        mkdir(foldername)
    end
    
    if ~exist(strcat(foldername, filename), 'file')
        fprintf('SPECT reconstruction saving to: %s%s\n', foldername, filename);

        %% Read in mumap, psf, proj1, proj2
        mumap = fld_read(mumap_path);
        mumap = mumap(:,:,mu_st:mu_end);
        
        psf = fld_read(psf_path);
        
        proj1 = fld_read(proj1_path);
        proj1 = proj1(:,sl_st:sl_end,:);
        assert(size(proj1, 3) == na*3, 'Error: Total number of views is not 120!!!');
        
        yi = proj1(:,:,1:120);
        r12 = proj1(:,:,121:240);
        r11 = proj1(:,:,241:360);
        ri = r11 + r12; % 128*82*120

        yi = yi(:,:,start:downfactor:end);
        ri = ri(:,:,start:downfactor:end);

        yi_pred = load(proj_nerf_path).results;
        ri_pred = load(scatter_nerf_path).results;
        yi_pred = imresize(yi_pred, 0.5, 'nearest');
        ri_pred = imresize(ri_pred, 0.5, 'nearest');
        yi_pred = yi_pred(:,sl_st:sl_end,:);
        ri_pred = ri_pred(:,sl_st:sl_end,:);
        yi_pred(yi_pred<0) = 0;
        ri_pred(ri_pred<0) = 0;
        yi_pred(isnan(yi_pred)) = 0;
        ri_pred(isnan(ri_pred)) = 0; 

        train_view_index = false(1,120);
        train_view_index(start:downfactor:end) = true;
        test_view_index = true(1,120);
        test_view_index(start:downfactor:end) = false;
        yi_tot(:,:,train_view_index) = yi;
        yi_tot(:,:,test_view_index) = yi_pred;
        ri_tot(:,:,train_view_index) = ri;
        ri_tot(:,:,test_view_index) = ri_pred;
        yi = yi_tot;
        ri = ri_tot;
        
        %% Recon
        ig = image_geom('nx', nx, 'ny', ny, 'nz', nz, 'dx', 4.7952, 'dz', 4.7952); % 128*128*82  % 82 if 82 slices
        ig.mask = ig.circ(ig.dx * (ig.nx/2-2), ig.dy * (ig.ny/2-1)) > 0;
        sg = sino_geom('par', 'nb', ig.nx, 'na', na, 'orbit_start', 360/na/2, 'orbit', 360, 'dr', ig.dx);
        f.dir = test_dir;
        f.file_mumap = [f.dir, 'mumap.fld'];
        fld_write(f.file_mumap, mumap); 
        f.file_psf = [f.dir, 'psf.fld'];
        fld_write(f.file_psf, psf); 
        f.sys_type = sprintf('3s@%g,%g,%g,%g,%g,1,fft@%s@%s@-%d,%d,%d', ig.dx, abs(ig.dy), abs(ig.dz), sg.orbit, sg.orbit_start, f.file_mumap, f.file_psf, sg.nb, ig.nz, sg.na);
        G = Gtomo3(f.sys_type, ig.mask, ig.nx, ig.ny, ig.nz, 'nthread', jf('ncore'), 'chat', 0);
        
        Gb = Gblock(G, num_block); 
        xinit = single(ig.mask); 
        
        ci = ones(size(yi));
        os_data = {reshaper(yi, '2d'), reshaper(ci, '2d'), ...
            reshaper(ri, '2d')};
        xosem = eml_osem(xinit(ig.mask), Gb, os_data{:}, ...
            'niter', niter);
        xosem = ig.embed(xosem);
        xosem(:,:,:,1) = [];
        
        free(G); 
        clear G Gb; % 

        fld_write(strcat(foldername, filename), xosem);
    
        %% Write the log file
        params_file = fullfile(foldername, 'paras.txt');
        fileID = fopen(params_file, 'w');
    
        if fileID == -1
            error('Cannot open file for writing: %s', params_file);
        end
    
        fprintf(fileID, 'nx = %d;\n', nx);
        fprintf(fileID, 'ny = %d;\n', ny);
        fprintf(fileID, 'nz = %d;\n', nz);
        fprintf(fileID, 'na = %d;\n', na);
        fprintf(fileID, 'niter = %d;\n', niter);
        fprintf(fileID, 'num_block = %d;\n', num_block);
        fprintf(fileID, 'mumap_path = %s;\n', mumap_path);
        fprintf(fileID, 'psf_path = %s;\n', psf_path);
        fprintf(fileID, 'proj1_path = %s;\n', proj1_path);
        fprintf(fileID, 'foldername = %s;\n', foldername);
        fprintf(fileID, 'filename = %s;\n', filename);
        fprintf(fileID, 'mumap = mumap(:,:,%d:%d);\n', mu_st, mu_end);
        fprintf(fileID, 'proj1 = proj1(:,%d:%d,:);\n', sl_st, sl_end);
    
        fclose(fileID);
    else
        fprintf('SPECT reconstruction: %s%s already exists!\n', foldername, filename);
    end
end
