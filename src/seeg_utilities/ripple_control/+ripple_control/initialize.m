function initialize(xippmex_directory)
    arguments
        xippmex_directory (1,1) string {mustBeFolder} = get_default_installation_directory()
    end

    addpath(xippmex_directory);

    % close existing XIPP session
    if xippmex
        % Stop XIPP thread, close UDP sockets, and clear caches/buffers.
        % Reference: Page 4 of Xippmex User Manual version LBL-0057_11
        xippmex('close');
    end

    % Initialize Xippmex in TCP mode to ensure that data
    % packets are reliably transmitted during the recording.
    % Reference: Page 3 of Xippmex User Manual version LBL-0057_11
    tcp_is_enabled = xippmex('tcp');
    if tcp_is_enabled
        fprintf('Xippmex was successfully initialized in TCP mode.\n');
    else
        error('Xippmex could not be initialized in TCP mode. Ensure that the Trellis GUI is configured to use TCP mode.\n');
    end
end

function xippmex_directory = get_default_installation_directory()
    if isunix
        xippmex_directory = "~/.local/share/Ripple/Trellis/Tools/xippmex";
    elseif ispc
        xippmex_directory = "C:\Program Files (x86)\Ripple\Trellis\Tools\xippmex";
    else
        error('Unsupported platform: must be UNIX or PC\n')
    end
end
