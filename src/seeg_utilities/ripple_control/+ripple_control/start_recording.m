function operator_id = start_recording(filepath_base)
    arguments
        filepath_base (1,1) string
    end

    operator_id = -1;
    recording_started_automatically = 0;

    try
        operator_id = input('Enter the Trellis operator ID [0-255, last octet of IP address found in the Network Info panel in the Trellis window]: ');

        % Required in TCP mode
        % Reference: Page 6 of Xippmex User Manual version LBL-0057_11
        xippmex('addoper', operator_id);

        [parent_directory, ~] = fileparts(filepath_base);
        if ~isfolder(parent_directory)
            mkdir(parent_directory);
            fprintf('Created directory %s\n', parent_directory);
        end

        recording_parameters_correct = input('Confirm that the recording parameters are correct on Trellis [1 for yes, 0 for no]: ');
        if ~recording_parameters_correct
            error('sEEG recording parameters are incorrect. Correct them on Trellis.\n')
        end

        trial_descriptor = xippmex('trial', 'recording', char(filepath_base), 0, 0, 0, operator_id);

        if strcmp(trial_descriptor.status, 'recording')
            recording_started_automatically = 1;
            fprintf('sEEG recording started automatically with output saved at %s\n', filepath_base);
        end
    catch
    end

    if ~recording_started_automatically
        warning('sEEG recording was not started automatically. Start the recording manually in the Trellis GUI now, using the filepath:\n%s\n', filepath_base);

        recording_started_manually = input('Confirm that the recording has been started manually [1 for yes, 0 for no]: ');
        if recording_started_manually
            fprintf('sEEG recording started manually!\n');
        else
            error('sEEG recording was not started automatically or manually.\n')
        end
    end
end
