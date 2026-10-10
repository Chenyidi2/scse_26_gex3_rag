#requires -Version 7.0
param(
    [string]$Python = 'python',
    [ValidateSet('index','verify')][string]$Task = 'verify',
    [ValidateSet('cuda','vulkan')][string]$Backend = 'cuda',
    [int]$VulkanDevice = 1,
    [int]$Port = 11436
)
$ErrorActionPreference='Stop'
$Python=(Get-Command $Python -ErrorAction Stop).Source
$ollama=(Get-Command ollama -ErrorAction Stop).Source
if(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue){throw 'Port is occupied. No existing process will be stopped.'}
function Resources {
    $os=Get-CimInstance Win32_OperatingSystem
    $raw=& nvidia-smi -i 0 --query-gpu=temperature.gpu,memory.free,memory.used --format=csv,noheader,nounits
    if($LASTEXITCODE -ne 0){throw 'GPU telemetry is unavailable.'}
    $values=$raw.Split(',') | ForEach-Object {[double]$_.Trim()}
    [pscustomobject]@{ram_gib=$os.FreePhysicalMemory/1MB;temperature=$values[0];vram_free_mib=$values[1];vram_used_mib=$values[2]}
}
$initial=Resources
if($initial.ram_gib -lt 6 -or $initial.vram_free_mib -lt 5120 -or $initial.temperature -gt 68){throw 'Need 6 GiB free RAM, 5 GiB free VRAM and GPU at most 68 C.'}
$folder=Join-Path $PSScriptRoot ('runs\'+(Get-Date -Format 'yyyyMMddTHHmmss')+'_'+$Task+'_'+[guid]::NewGuid().ToString('N').Substring(0,6))
$null=New-Item -ItemType Directory -Path $folder
$settings=@{
    OLLAMA_HOST="127.0.0.1:$Port";OLLAMA_BASE_URL="http://127.0.0.1:$Port"
    OLLAMA_KEEP_ALIVE='2m';OLLAMA_NUM_PARALLEL='1';OLLAMA_MAX_LOADED_MODELS='1'
    OLLAMA_NO_CLOUD='1';OLLAMA_NOPRUNE='1';OLLAMA_DEBUG='1'
    OMP_NUM_THREADS='2';MKL_NUM_THREADS='2';OPENBLAS_NUM_THREADS='2'
    ANONYMIZED_TELEMETRY='False';PYTHONUNBUFFERED='1'
}
if($Backend -eq 'vulkan'){
    $settings.OLLAMA_LLM_LIBRARY='vulkan';$settings.OLLAMA_VULKAN='1'
    $settings.CUDA_VISIBLE_DEVICES='-1';$settings.GGML_VK_VISIBLE_DEVICES=[string]$VulkanDevice
}else{
    $settings.OLLAMA_LLM_LIBRARY=$null;$settings.OLLAMA_VULKAN=$null
    $settings.CUDA_VISIBLE_DEVICES='0';$settings.GGML_VK_VISIBLE_DEVICES=$null
}
$previous=@{};$server=$null;$worker=$null
$samples=[Collections.Generic.List[object]]::new()
$record=[ordered]@{status='starting';task=$Task;backend=$Backend;started=(Get-Date).ToString('o');initial=$initial;port=$Port}
$timer=[Diagnostics.Stopwatch]::StartNew()
try {
    foreach($key in $settings.Keys){$previous[$key]=[Environment]::GetEnvironmentVariable($key,'Process');[Environment]::SetEnvironmentVariable($key,$settings[$key],'Process')}
    $server=Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'server.out.log') -RedirectStandardError (Join-Path $folder 'server.err.log')
    $record.server_pid=$server.Id
    $ready=$false
    for($i=0;$i -lt 30;$i++){
        if($server.HasExited){throw 'Temporary model service stopped during startup.'}
        try{$null=Invoke-RestMethod "http://127.0.0.1:$Port/api/version" -NoProxy -TimeoutSec 2;$ready=$true;break}catch{Start-Sleep -Seconds 1}
    }
    if(-not $ready){throw 'Temporary model service did not become ready.'}
    $models=(Invoke-RestMethod "http://127.0.0.1:$Port/api/tags" -NoProxy -TimeoutSec 10).models.name
    foreach($name in @('nomic-embed-text:latest','qwen3:0.6b')){
        if($name -notin $models){throw "Missing local model: $name. Pull it before running this script."}
    }
    $arguments=if($Task -eq 'index'){@('-B','-u','rag.py')}else{@('-B','-u','verify_live.py','--output',('"'+(Join-Path $folder 'answers.json')+'"'))}
    $worker=Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'worker.out.log') -RedirectStandardError (Join-Path $folder 'worker.err.log')
    $record.worker_pid=$worker.Id
    Write-Output "Run record: $folder"
    while(-not $worker.HasExited){
        $sample=Resources
        $samples.Add([pscustomobject]@{seconds=$timer.Elapsed.TotalSeconds;resources=$sample})
        if($sample.ram_gib -lt 3 -or $sample.temperature -ge 82 -or $sample.vram_free_mib -lt 256){throw 'Resource guard stopped this run.'}
        if($server.HasExited){throw 'Temporary model service stopped unexpectedly.'}
        if($timer.Elapsed.TotalSeconds -gt 1200){throw 'Run exceeded 20 minutes.'}
        Start-Sleep -Seconds 2
    }
    $worker.WaitForExit();$record.worker_exit=$worker.ExitCode
    if($worker.ExitCode -ne 0){throw 'Run failed; logs retained.'}
    $record.status='passed'
}catch{$record.status='failed';$record.error=$_.Exception.Message;Write-Output $record.error}
finally{
    foreach($process in @($worker,$server)){
        if($process -and -not $process.HasExited){$process.Kill($true);$null=$process.WaitForExit(10000)}
    }
    foreach($key in $previous.Keys){[Environment]::SetEnvironmentVariable($key,$previous[$key],'Process')}
    $record.finished=(Get-Date).ToString('o');$record.samples=$samples.ToArray()
    $record.service_stopped=-not [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $folder 'monitor.json') -Encoding utf8
    if($worker){Get-Content -LiteralPath (Join-Path $folder 'worker.out.log') -Tail 8;Get-Content -LiteralPath (Join-Path $folder 'worker.err.log') -Tail 8}
    Write-Output "Status: $($record.status); temporary service stopped: $($record.service_stopped)"
}
if($record.status -ne 'passed' -or -not $record.service_stopped){exit 1}
