package cn.vabpnn.mod.client.capture;

import com.mojang.blaze3d.systems.RenderSystem;
import java.nio.ByteBuffer;

import static org.lwjgl.opengl.GL33C.*;

/** One owned RGBA8 texture/FBO slot. No CPU readback path exists. */
public final class OpenGlCaptureSurface implements AutoCloseable {
    private int framebuffer;
    private int texture;
    private int width;
    private int height;

    public int texture() { return texture; }

    public void prepare(int newWidth, int newHeight) {
        RenderSystem.assertOnRenderThread();
        if (framebuffer != 0) throw new IllegalStateException("Retire CUDA ownership before replacing a texture");
        int previousDraw = glGetInteger(GL_DRAW_FRAMEBUFFER_BINDING);
        int previousTexture = glGetInteger(GL_TEXTURE_BINDING_2D);
        int previousUnpackBuffer = glGetInteger(GL_PIXEL_UNPACK_BUFFER_BINDING);
        try {
            texture = glGenTextures();
            glBindTexture(GL_TEXTURE_2D, texture);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, newWidth, newHeight, 0,
                    GL_RGBA, GL_UNSIGNED_BYTE, (ByteBuffer) null);
            framebuffer = glGenFramebuffers();
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
            glDrawBuffer(GL_COLOR_ATTACHMENT0);
            int status = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
            if (status != GL_FRAMEBUFFER_COMPLETE) {
                throw new IllegalStateException("Incomplete capture FBO: 0x" + Integer.toHexString(status));
            }
            width = newWidth;
            height = newHeight;
        } catch (RuntimeException | OutOfMemoryError error) {
            close();
            throw error;
        } finally {
            glBindTexture(GL_TEXTURE_2D, previousTexture);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, previousUnpackBuffer);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previousDraw);
        }
    }

    public void copyPresentedFrame() {
        RenderSystem.assertOnRenderThread();
        int previousRead = glGetInteger(GL_READ_FRAMEBUFFER_BINDING);
        int previousDraw = glGetInteger(GL_DRAW_FRAMEBUFFER_BINDING);
        boolean scissor = glIsEnabled(GL_SCISSOR_TEST);
        boolean srgb = glIsEnabled(GL_FRAMEBUFFER_SRGB);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        int defaultReadBuffer = glGetInteger(GL_READ_BUFFER);
        try {
            glReadBuffer(glGetBoolean(GL_DOUBLEBUFFER) ? GL_BACK : GL_FRONT);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_FRAMEBUFFER_SRGB);
            glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                    GL_COLOR_BUFFER_BIT, GL_NEAREST);
        } finally {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            glReadBuffer(defaultReadBuffer);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, previousRead);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previousDraw);
            if (scissor) glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
            if (srgb) glEnable(GL_FRAMEBUFFER_SRGB); else glDisable(GL_FRAMEBUFFER_SRGB);
        }
    }

    @Override public void close() {
        RenderSystem.assertOnRenderThread();
        if (framebuffer != 0) glDeleteFramebuffers(framebuffer);
        if (texture != 0) glDeleteTextures(texture);
        framebuffer = texture = width = height = 0;
    }
}
