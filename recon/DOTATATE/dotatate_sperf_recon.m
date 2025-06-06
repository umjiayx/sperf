function dotatate_sperf_recon(DOTATATE_patient_id)
    nx = 128;
    ny = 128;
    nz = 81;
    na = 120;
    niter = 20;
    num_block = 6;
    sl_st = 24; 
    sl_end = 104;
    downfactor = 4;
    start = 1;

    proj_path = strcat('./proj/proj_', DOTATATE_patient_id, '.fld');
    mumap_path = strcat('./mumap/mumap_', DOTATATE_patient_id, '.fld');
    psf_path = strcat('./psf/psf_proj_', DOTATATE_patient_id, '_208.fld');
    
    proj = fld_read(proj_path);
    mumap = fld_read(mumap_path);
    psf = fld_read(psf_path);

    foldername = strcat('./recon/sperfrecon/', DOTATATE_patient_id, '/');
    filename_20iters = 'sperfrecon_81_20iters.fld';
    filename = 'sperfrecon_81.fld';
    pred_proj_path = strcat('./proj_sperf_mat/', DOTATATE_patient_id, '/test/pred_proj_test.mat');
    pred_scatter_path = strcat('./proj_sperf_mat/', DOTATATE_patient_id, '_scatter/test/pred_proj_test.mat');

    if ~exist(foldername, 'dir')
        mkdir(foldername)
    end

    if ~exist(strcat(foldername, filename), 'file')
        fprintf('SPECT reconstruction saving to: %s%s\n', foldername, filename);

        proj = proj(:,sl_st:sl_end,:); 
        yi = proj(:,:,1:120); 
        ri2 = proj(:,:,121:240); 
        ri1 = proj(:,:,241:360);
        ri = ri1+ri2;
        yi = yi(:,:,start:downfactor:end);
        ri = ri(:,:,start:downfactor:end);
        yi_pred = load(pred_proj_path).results;
        ri_pred = load(pred_scatter_path).results;

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
        ig = image_geom('nx', nx, 'ny', ny, 'nz', nz, 'dx', 4.7952, 'dz', 4.7952);
        ig.mask = ig.circ(ig.dx * (ig.nx/2-2), ig.dy * (ig.ny/2-1)) > 0;
        sg = sino_geom('par', 'nb', ig.nx, 'na', na, 'orbit_start', 360/2/(size(proj, 3)/3), 'orbit', 360, 'dr', ig.dx);
        f.dir = test_dir;
        f.file_mumap = [f.dir, 'mumap.fld'];
        fld_write(f.file_mumap, mumap); % save mu map to file
        f.file_psf = [f.dir, 'psf.fld'];
        fld_write(f.file_psf, psf); % save psf to file
        f.sys_type = sprintf('3s@%g,%g,%g,%g,%g,1,fft@%s@%s@-%d,%d,%d', ig.dx, abs(ig.dy), abs(ig.dz), sg.orbit, sg.orbit_start, f.file_mumap, f.file_psf, sg.nb, ig.nz, sg.na);
        G = Gtomo3(f.sys_type, ig.mask, ig.nx, ig.ny, ig.nz, 'nthread', jf('ncore'), 'chat', 0);
        
        Gb = Gblock(G, num_block); 
        xinit = single(ig.mask); % uniform
        
        % no scatter correction
        ci = ones(size(yi));
        os_data = {reshaper(yi, '2d'), reshaper(ci, '2d'), ...
            reshaper(ri, '2d')};
        xosem = eml_osem(xinit(ig.mask), Gb, os_data{:}, ...
            'niter', niter);
        xosem = ig.embed(xosem);
        xosem(:,:,:,1) = [];
        
        free(G); 
        clear G Gb; % 
        fld_write(strcat(foldername, filename_20iters), xosem, 'type', 'xdr_float');
        fld_write(strcat(foldername, filename), xosem(:,:,:,10), 'type', 'xdr_float');
    else
        fprintf('SPECT reconstruction: %s%s already exists!\n', foldername, filename);
    end
end