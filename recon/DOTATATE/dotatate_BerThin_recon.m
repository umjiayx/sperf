% This script is for SPECT reconstruction for 1/d time per angle, where 
% Bernoulli thinning method is applied to full-time projections. 

function dotatate_BerThin_recon(DOTATATE_patient_id)
    %%
    DOTATATE_patient_id = "patient1_c1_s2";
    nx = 128;
    ny = 128;
    nz = 81;
    na = 120;
    niter = 20;
    num_block = 6;
    sl_st = 24; 
    sl_end = 104;
    downfactor = 2;
    % start = 1;

    % proj_path = strcat('./proj/proj_', DOTATATE_patient_id, '.fld');
    mumap_path = strcat('./mumap/mumap_', DOTATATE_patient_id, '.fld');
    psf_path = strcat('./psf/psf_proj_', DOTATATE_patient_id, '_208.fld');
    
    % proj = fld_read(proj_path);
    mumap = fld_read(mumap_path);
    psf = fld_read(psf_path);

    foldername = strcat('./recon_d=', int2str(downfactor), '/berthirecon/', DOTATATE_patient_id, '/');
    filename_20iters = 'berthirecon_81_20iters.fld';
    filename = 'berthirecon_81.fld';
    
    % DOTATATE_patient_id_short = regexp(DOTATATE_patient_id, '^patient\d+', 'match', 'once');
    proj1_path = strcat("./proj/projs_bernoulli_thinning_scan/proj_", DOTATATE_patient_id, "_d=", int2str(downfactor), ".fld");
    scatter1_path = strcat("./proj/scatters_bernoulli_thinning_scan/proj_", DOTATATE_patient_id, "_scatter_d=", int2str(downfactor), ".fld");

    if ~exist(foldername, 'dir')
        mkdir(foldername)
    end
    
    %% 
    if ~exist(strcat(foldername, filename), 'file')
        fprintf('SPECT reconstruction saving to: %s%s\n', foldername, filename);
        % Load proj1 and scatter1
        proj1 = fld_read(proj1_path);
        proj1 = proj1(:, sl_st:sl_end, :); 
        scatter1 = fld_read(scatter1_path);
        scatter1 = scatter1(:, sl_st:sl_end, :);

        yi = proj1;
        ri = scatter1;

        %% Recon
        ig = image_geom('nx', nx, 'ny', ny, 'nz', nz, 'dx', 4.7952, 'dz', 4.7952);
        ig.mask = ig.circ(ig.dx * (ig.nx/2-2), ig.dy * (ig.ny/2-1)) > 0;
        sg = sino_geom('par', 'nb', ig.nx, 'na', na, 'orbit_start', 360/2/(size(yi, 3)), 'orbit', 360, 'dr', ig.dx);
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