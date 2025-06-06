function psma_full_recon_bed1(patient_id, patient_name)
    %% Parameters
    nx = 128;
    ny = 128;
    nz = 79;
    na = 120;
    niter = 20;
    num_block = 6;
    sl_st = 24;
    sl_end = 102;
    mu_st = 1;
    mu_end = 79;    

    mumap_path = ['./mumap_fld/mumap_psma_test_', patient_name, '.fld'];
    psf_path = ['./psf/psf/psf_proj_psma_', patient_id, '_b1_208.fld'];
    proj1_path = ['./proj_fld/proj_psma_', patient_id, '_b1.fld'];
    foldername = ['./recon/full_recon/psma_', patient_id, '_b1/'];
    filename = 'fullrecon_79slices.fld';

    if ~exist(foldername, 'dir')
        mkdir(foldername)
    end
    
    if ~exist(strcat(foldername, filename), 'file')
        fprintf('SPECT reconstruction saving to: %s%s\n', foldername, filename);

        %% Read in mumap, psf, proj1, proj2
        mumap = fld_read(mumap_path);
        mumap = mumap(:,:,mu_st:mu_end); % If you change this line, please change the log!!!!
        
        psf = fld_read(psf_path);
        
        proj1 = fld_read(proj1_path);
        proj1 = proj1(:,sl_st:sl_end,:); % If you change this line, please change the log!!!!
        assert(size(proj1, 3) == na*3, 'Error: Total number of views is not 120!!!');
        
        yi = proj1(:,:,1:120);
        r12 = proj1(:,:,121:240);
        r11 = proj1(:,:,241:360);
        ri = r11 + r12; % 128*82*120
        
        %% Recon
        ig = image_geom('nx', nx, 'ny', ny, 'nz', nz, 'dx', 4.7952, 'dz', 4.7952); % 128*128*82  % 82 if 82 slices
        ig.mask = ig.circ(ig.dx * (ig.nx/2-2), ig.dy * (ig.ny/2-1)) > 0;
        sg = sino_geom('par', 'nb', ig.nx, 'na', na, 'orbit_start', 1.5, 'orbit', 360, 'dr', ig.dx);
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
    end


end
