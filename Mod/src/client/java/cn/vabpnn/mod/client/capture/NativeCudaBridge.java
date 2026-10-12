package cn.vabpnn.mod.client.capture;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/** JNI control only; pixels and tensors never cross into Java/CPU memory. */
public final class NativeCudaBridge implements AutoCloseable {
    public static final int INITIALIZING = 0, READY = 1, RETIRING = 2, RETIRED = 3;
    private static boolean loaded;
    private long handle;
    private final long[] requestedSave=new long[3];

    public record Stats(long submitted, long copied, long processed, long skipped, long dropped,
                        long lastProcessed, long memoryBytes, long registrations, int capacity, int phase,
                        long modelBytes) {}

    public NativeCudaBridge(int capacity) {
        loadLibrary();
        handle = nStart(capacity);
    }

    private static synchronized void loadLibrary() {
        if (loaded) return;
        if (!System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("windows")
                || !System.getProperty("os.arch").equals("amd64")) {
            throw new IllegalStateException("This build requires Windows x64 and an NVIDIA OpenGL/CUDA GPU");
        }
        String override = System.getProperty("vabpnn.cuda.library");
        if (override != null && !override.isBlank()) {
            System.load(Path.of(override).toAbsolutePath().toString());
        } else {
            // Only the executable native library is extracted. No frame/image data is written.
            try (var resource = NativeCudaBridge.class.getResourceAsStream(
                    "/native/windows-x86_64/vabpnn_capture.dll")) {
                if (resource == null) throw new IllegalStateException("CUDA DLL is missing from the mod jar");
                Path library = Files.createTempFile("vabpnn-cuda-", ".dll");
                Files.copy(resource, library, StandardCopyOption.REPLACE_EXISTING);
                library.toFile().deleteOnExit();
                System.load(library.toString());
            } catch (IOException error) {
                throw new IllegalStateException("Could not load native CUDA library", error);
            }
        }
        loaded = true;
    }

    public void configure(int[] textures, int width, int height) { nConfigure(handle, textures, width, height); }
    public int acquire() { return nAcquire(handle); }
    public boolean submit(int slot, long fence, long frameId, long captureNanos) {
        return nSubmit(handle, slot, fence, frameId, captureNanos);
    }
    /** Final command only: frame/time/mask/valid and three float bit patterns; buffer is reused. */
    public void readControl(long[] command) { nControl(handle, command); }
    public void demonstration(long[] values, double[] motion) { nDemo(handle, values, motion); }
    /** At most 30 Hz; only timestamp/epoch/XYZ. The worker attaches it to the submitted frame. */
    public void position(long nanos,long epoch,double x,double y,double z) { nPosition(handle,nanos,epoch,x,y,z); }
    /** Completed frame's coordinate timestamp/epoch/AdamW steps/validity/XYZ double bit patterns. */
    public void readPosition(long[] status) { nPositionStatus(handle,status); }
    /** Elapsed minutes (double bits), observed minute buckets, minute AdamW steps, frame and Bayesian mouse steps. */
    public void readTime(long[] status) { nTimeStatus(handle,status); }
    public void checkpoint(Path path) { nCheckpoint(handle, path.toAbsolutePath().toString(), false); }
    /** Returns a request serial; completion is published only after the worker finishes saving. */
    public long saveCheckpoint() {
        nCheckpoint(handle, null, true);
        readSaveStatus(requestedSave);
        return requestedSave[0];
    }
    /** Request serial, finished serial, checkpoint result; all are CPU metadata. */
    public void readSaveStatus(long[] status) { nSaveStatus(handle, status); }
    public void readTraining(long[] status) { nTraining(handle, status); }
    /** Completed GPU frame, validity, eligible mask, learned k, selected count and last human label. */
    public void readDecision(long[] status) { nDecision(handle,status); }
    public void configureBlocks(int enabled,int communication) { nBlocks(handle,enabled,communication); }
    public void readBlocks(long[] status) { nBlockStatus(handle,status); }
    public void configureMemes(int enabled) { nMemes(handle,enabled); }
    public void readMemes(long[] status) { nMemeStatus(handle,status); }
    public void readTiming(long[] status) { nTiming(handle, status); }
    public String checkpointMessage() { return nCheckpointText(handle); }
    public void resetDecision(long minimumFrame) { nResetDecision(handle, minimumFrame); }

    public void memeScopeEnabled(boolean enabled,int tile) { nMemeScopeEnabled(handle,enabled,tile); }
    public void readMemeScope(long[] meta,float[] values,int[] discrete,long[] identities) {
        nMemeScope(handle,meta,values,discrete,identities);
    }
    public void scopeEnabled(boolean enabled,int tile) { nScopeEnabled(handle,enabled,tile); }
    public void readScope(long[] meta,float[] values,int[] discrete) { nScope(handle,meta,values,discrete); }

    public void cancel(int slot) { nCancel(handle, slot); }
    public void retire() { nRetire(handle); }
    public int phase() { return nPhase(handle); }
    public String error() { return nText(handle, true); }
    public String deviceName() { return nText(handle, false); }
    public Stats stats() {
        long[] v = nStats(handle);
        return new Stats(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], (int) v[8], (int) v[9], v[10]);
    }

    @Override public void close() {
        if (handle != 0) {
            nClose(handle);
            handle = 0;
        }
    }

    private static native void nMemeScopeEnabled(long handle,boolean enabled,int tile);
    private static native void nMemeScope(long handle,long[] meta,float[] values,int[] discrete,long[] identities);
    private static native void nScopeEnabled(long handle,boolean enabled,int tile);
    private static native void nScope(long handle,long[] meta,float[] values,int[] discrete);
    private static native void nBlocks(long handle,int enabled,int communication);
    private static native void nBlockStatus(long handle,long[] status);
    private static native void nMemes(long handle,int enabled);
    private static native void nMemeStatus(long handle,long[] status);
    private static native void nTiming(long handle, long[] status);
    private static native void nDemo(long handle, long[] values, double[] motion);
    private static native void nPosition(long handle,long nanos,long epoch,double x,double y,double z);
    private static native void nPositionStatus(long handle,long[] status);
    private static native void nTimeStatus(long handle,long[] status);
    private static native void nCheckpoint(long handle, String path, boolean save);
    private static native void nTraining(long handle, long[] status);
    private static native void nDecision(long handle,long[] status);
    private static native void nSaveStatus(long handle, long[] status);
    private static native String nCheckpointText(long handle);
    private static native long nStart(int capacity);
    private static native void nConfigure(long handle, int[] textures, int width, int height);
    private static native int nAcquire(long handle);
    private static native boolean nSubmit(long handle, int slot, long fence, long frameId, long captureNanos);
    private static native void nCancel(long handle, int slot);
    private static native void nRetire(long handle);
    private static native int nPhase(long handle);
    private static native long[] nStats(long handle);
    private static native void nControl(long handle, long[] command);
    private static native void nResetDecision(long handle, long minimumFrame);
    private static native String nText(long handle, boolean error);
    private static native void nClose(long handle);
}


