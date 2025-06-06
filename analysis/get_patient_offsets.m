function patients_offsets = get_patient_offsets()
    % Patient offsets for interpolating (data acquired from the scanner)
    patients_offsets = struct([]);

    patients_offsets(1).Y = -0.25;
    patients_offsets(1).X = 0.66;
    patients_offsets(1).Z = -4.98;

    patients_offsets(2).Y = -0.25;
    patients_offsets(2).X = 0.57;
    patients_offsets(2).Z = -2.49;

    patients_offsets(3).Y = -0.23;
    patients_offsets(3).X = 1.66;
    patients_offsets(3).Z = -3.66;
    
    patients_offsets(4).Y = -0.23;
    patients_offsets(4).X = 1.73;
    patients_offsets(4).Z = -4.7;
    
    patients_offsets(5).Y = -0.23;
    patients_offsets(5).X = 1.87;
    patients_offsets(5).Z = -3.68;
    
    patients_offsets(6).Y = -0.23;
    patients_offsets(6).X = 1.67;
    patients_offsets(6).Z = -0.64;
    
    patients_offsets(7).Y = 0;
    patients_offsets(7).X = 0.4;
    patients_offsets(7).Z = -2.11;
    
    patients_offsets(8).Y = -0.23;
    patients_offsets(8).X = 1.31;
    patients_offsets(8).Z = -1.77;
    
    patients_offsets(9).Y = -0.23;
    patients_offsets(9).X = 1.82;
    patients_offsets(9).Z = -3.33;
    
    patients_offsets(10).Y = -0.23;
    patients_offsets(10).X = 1.2;
    patients_offsets(10).Z = -3.41;
    
    patients_offsets(11).Y = -0.23;
    patients_offsets(11).X = 1.05;
    patients_offsets(11).Z = -1.47;
end