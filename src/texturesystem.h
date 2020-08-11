#pragma once

#include <OpenImageIO/texture.h>

class TextureSystem {
    public:
    static void Init() {
        s_TextureSystem = OIIO::TextureSystem::create();
        s_TextureSystem->attribute("autotile", 64);
        s_TextureSystem->attribute("automip", 1);
        s_TextureSystem->attribute("forcefloat", 1);
    }
    static void Destroy() {
        OIIO::TextureSystem::destroy(s_TextureSystem);
    }

    private:
    static OIIO::TextureSystem *s_TextureSystem;
    template <int nChannels>
    friend class BitmapTexture;
};
